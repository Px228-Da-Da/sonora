//! Per-application capture via WASAPI process loopback.
//!
//! Windows can hand back exactly what one process (and its children) is playing,
//! through `ActivateAudioInterfaceAsync` on the pseudo-device
//! `VAD\Process_Loopback`. cpal has no API for it, so this module talks to WASAPI
//! directly. Needs Windows 10 build 20348 or later.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use windows::core::{implement, Interface, Ref, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioCaptureClient, IAudioClient, IAudioSessionControl2,
    IAudioSessionManager2, IActivateAudioInterfaceAsyncOperation,
    IActivateAudioInterfaceCompletionHandler, IActivateAudioInterfaceCompletionHandler_Impl,
    IMMDeviceEnumerator, MMDeviceEnumerator, ActivateAudioInterfaceAsync,
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_LOOPBACK, AUDIOCLIENT_ACTIVATION_PARAMS,
    AUDIOCLIENT_ACTIVATION_PARAMS_0, AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
    AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, WAVEFORMATEX,
};
use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL,
    COINIT_MULTITHREADED,
};
use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
use windows::Win32::System::Threading::{
    CreateEventW, OpenProcess, QueryFullProcessImageNameW, SetEvent, WaitForSingleObject,
    PROCESS_NAME_FORMAT, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::System::Variant::VT_BLOB;

use crate::audio::SharedAudio;

/// Process loopback does not negotiate a format — the caller states one and WASAPI
/// converts. Stereo float at 48 kHz matches what the rest of the app expects.
const CAPTURE_RATE: u32 = 48_000;
const CAPTURE_CHANNELS: u16 = 2;
/// `WAVE_FORMAT_IEEE_FLOAT`; the binding lives in a module we don't otherwise need.
const FORMAT_IEEE_FLOAT: u16 = 3;

/// An application currently holding an audio session on the default output.
pub struct AppEntry {
    pub pid: u32,
    pub name: String,
}

/// COM must be initialised per thread. A thread that already has it (winit does this
/// on the UI thread) reports `S_FALSE` or `RPC_E_CHANGED_MODE`; both are fine to work
/// with, we simply must not uninitialise what we did not initialise.
struct ComGuard {
    owned: bool,
}

impl ComGuard {
    fn new() -> Self {
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        Self { owned: hr.is_ok() }
    }
}

impl Drop for ComGuard {
    fn drop(&mut self) {
        if self.owned {
            unsafe { CoUninitialize() };
        }
    }
}

/// Executable name for a pid, without the path.
fn process_name(pid: u32) -> Option<String> {
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buf = [0u16; 260];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_FORMAT(0),
            windows::core::PWSTR(buf.as_mut_ptr()),
            &mut len,
        );
        let _ = CloseHandle(handle);
        ok.ok()?;
        let full = String::from_utf16_lossy(&buf[..len as usize]);
        Some(full.rsplit('\\').next().unwrap_or(&full).to_string())
    }
}

/// Applications with an audio session on the default output device, newest first.
///
/// Sessions linger in an inactive state for a while after a sound stops, so this
/// lists more than what is audible right now — which is what you want when picking
/// a target before starting playback.
pub fn list_apps() -> Vec<AppEntry> {
    let mut out: Vec<AppEntry> = Vec::new();
    let _com = ComGuard::new();
    unsafe {
        let Ok(enumerator) =
            CoCreateInstance::<_, IMMDeviceEnumerator>(&MMDeviceEnumerator, None, CLSCTX_ALL)
        else {
            return out;
        };
        let Ok(device) = enumerator.GetDefaultAudioEndpoint(eRender, eConsole) else {
            return out;
        };
        let Ok(manager) = device.Activate::<IAudioSessionManager2>(CLSCTX_ALL, None) else {
            return out;
        };
        let Ok(sessions) = manager.GetSessionEnumerator() else {
            return out;
        };
        let count = sessions.GetCount().unwrap_or(0);
        for i in 0..count {
            let Ok(ctrl) = sessions.GetSession(i) else {
                continue;
            };
            let Ok(ctrl2) = ctrl.cast::<IAudioSessionControl2>() else {
                continue;
            };
            // pid 0 is the system-sounds session, which has no process to target.
            let Ok(pid) = ctrl2.GetProcessId() else {
                continue;
            };
            // pid 0 is the system-sounds session; our own loopback capture also
            // shows up as a session, and targeting it would capture nothing.
            if pid == 0 || pid == std::process::id() || out.iter().any(|a| a.pid == pid) {
                continue;
            }
            if let Some(name) = process_name(pid) {
                out.push(AppEntry { pid, name });
            }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

/// Keep a refreshed list of audio apps without touching COM on the UI thread.
///
/// Enumerating sessions is a handful of COM round-trips; doing it per frame would be
/// wasteful, and the UI thread runs COM in a different apartment mode anyway.
pub fn spawn_app_watcher() -> Arc<Mutex<Vec<AppEntry>>> {
    let list = Arc::new(Mutex::new(Vec::new()));
    let out = list.clone();
    std::thread::Builder::new()
        .name("sonora-app-list".into())
        .spawn(move || loop {
            let apps = list_apps();
            match list.lock() {
                Ok(mut slot) => *slot = apps,
                Err(_) => return,
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        })
        .ok();
    out
}

/// Linear gain the endpoint's volume control is currently applying, or `None` when
/// it can't be read. Muted reads as `0.0`.
///
/// Shared-mode loopback captures the mix *after* the endpoint volume, so the meters
/// follow the Windows slider unless this is divided back out.
pub fn endpoint_gain() -> Option<f32> {
    let _com = ComGuard::new();
    unsafe {
        let enumerator =
            CoCreateInstance::<_, IMMDeviceEnumerator>(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .ok()?;
        let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole).ok()?;
        let vol = device.Activate::<IAudioEndpointVolume>(CLSCTX_ALL, None).ok()?;
        if vol.GetMute().ok()?.as_bool() {
            return Some(0.0);
        }
        // The attenuation in dB is what the mixer actually applies; the "scalar" is
        // the slider position after Windows' taper and is not the same number.
        let db = vol.GetMasterVolumeLevel().ok()?;
        Some(10f32.powf(db / 20.0))
    }
}

/// Poll the endpoint volume off the UI thread; it changes rarely and reading it is
/// several COM calls.
pub fn spawn_volume_watcher() -> Arc<Mutex<Option<f32>>> {
    let gain = Arc::new(Mutex::new(endpoint_gain()));
    let out = gain.clone();
    std::thread::Builder::new()
        .name("sonora-volume".into())
        .spawn(move || loop {
            let g = endpoint_gain();
            match gain.lock() {
                Ok(mut slot) => *slot = g,
                Err(_) => return,
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        })
        .ok();
    out
}

/// Signals the async activation is done. `ActivateAudioInterfaceAsync` returns
/// immediately and calls this back, so the caller waits on the event.
#[implement(IActivateAudioInterfaceCompletionHandler)]
struct ActivateHandler(HANDLE);

impl IActivateAudioInterfaceCompletionHandler_Impl for ActivateHandler_Impl {
    fn ActivateCompleted(
        &self,
        _op: Ref<'_, IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        unsafe { SetEvent(self.0) }
    }
}

/// A running per-process capture. Dropping it stops the thread.
pub struct ProcessCapture {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for ProcessCapture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Start capturing everything `pid` and its children play.
///
/// COM objects are not `Send`, so the whole pipeline is built and torn down on the
/// capture thread; this call blocks only until that thread reports success.
pub fn start(
    pid: u32,
    shared: Arc<Mutex<SharedAudio>>,
    errors: Arc<Mutex<Option<String>>>,
) -> Result<(ProcessCapture, u32, u16), String> {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Result<(), String>>();

    let thread_stop = stop.clone();
    let thread = std::thread::Builder::new()
        .name("sonora-proc-capture".into())
        .spawn(move || {
            let result = capture_loop(pid, &shared, &thread_stop, &tx);
            if let Err(e) = result {
                // If the failure happened after start-up, the UI polls for it.
                let _ = tx.send(Err(e.clone()));
                if let Ok(mut slot) = errors.lock() {
                    *slot = Some(e);
                }
            }
        })
        .map_err(|e| format!("не удалось запустить поток захвата: {e}"))?;

    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(Ok(())) => Ok((
            ProcessCapture {
                stop,
                thread: Some(thread),
            },
            CAPTURE_RATE,
            CAPTURE_CHANNELS,
        )),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("захват процесса не ответил".to_string()),
    }
}

fn capture_loop(
    pid: u32,
    shared: &Arc<Mutex<SharedAudio>>,
    stop: &AtomicBool,
    ready: &mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let _com = ComGuard::new();

    unsafe {
        let done_event =
            CreateEventW(None, false, false, PCWSTR::null()).map_err(|e| format!("event: {e}"))?;
        let buffer_event =
            CreateEventW(None, false, false, PCWSTR::null()).map_err(|e| format!("event: {e}"))?;

        let mut params = AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                    TargetProcessId: pid,
                    ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                },
            },
        };
        // The activation parameters travel as a VT_BLOB pointing at the struct above.
        let raw = BlobPropVariant {
            vt: VT_BLOB.0,
            reserved: [0; 3],
            cb_size: std::mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
            pad: 0,
            p_blob_data: &mut params as *mut _ as *mut u8,
        };
        let prop = &*(&raw as *const BlobPropVariant as *const PROPVARIANT);

        let handler: IActivateAudioInterfaceCompletionHandler =
            ActivateHandler(done_event).into();
        let op: IActivateAudioInterfaceAsyncOperation = ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some(prop),
            &handler,
        )
        .map_err(|e| format!("ActivateAudioInterfaceAsync: {e}"))?;

        if WaitForSingleObject(done_event, 5000) != WAIT_OBJECT_0 {
            return Err("активация process loopback не завершилась".into());
        }
        let mut hr = windows::core::HRESULT(0);
        let mut unknown = None;
        op.GetActivateResult(&mut hr, &mut unknown)
            .map_err(|e| format!("GetActivateResult: {e}"))?;
        hr.ok().map_err(|e| {
            format!("process loopback недоступен ({e}) — нужна Windows 10 сборки 20348 или новее")
        })?;
        let client: IAudioClient = unknown
            .ok_or_else(|| "GetActivateResult вернул пусто".to_string())?
            .cast()
            .map_err(|e| format!("cast IAudioClient: {e}"))?;

        // Process loopback cannot negotiate: GetMixFormat is unsupported here, the
        // format below is what WASAPI converts into.
        let block_align = CAPTURE_CHANNELS * 4;
        let format = WAVEFORMATEX {
            wFormatTag: FORMAT_IEEE_FLOAT,
            nChannels: CAPTURE_CHANNELS,
            nSamplesPerSec: CAPTURE_RATE,
            nAvgBytesPerSec: CAPTURE_RATE * block_align as u32,
            nBlockAlign: block_align,
            wBitsPerSample: 32,
            cbSize: 0,
        };
        client
            .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                2_000_000, // 200 ms, in 100 ns units
                0,
                &format,
                None,
            )
            .map_err(|e| format!("IAudioClient::Initialize: {e}"))?;
        client
            .SetEventHandle(buffer_event)
            .map_err(|e| format!("SetEventHandle: {e}"))?;
        let capture: IAudioCaptureClient = client
            .GetService()
            .map_err(|e| format!("GetService(IAudioCaptureClient): {e}"))?;
        client.Start().map_err(|e| format!("Start: {e}"))?;

        let _ = ready.send(Ok(()));

        while !stop.load(Ordering::Relaxed) {
            if WaitForSingleObject(buffer_event, 200) != WAIT_OBJECT_0 {
                continue; // timeout: the app may simply be silent
            }
            loop {
                let mut data: *mut u8 = std::ptr::null_mut();
                let mut frames = 0u32;
                let mut flags = 0u32;
                if capture
                    .GetBuffer(&mut data, &mut frames, &mut flags, None, None)
                    .is_err()
                    || frames == 0
                {
                    break;
                }
                let silent = flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0;
                let mut out = Vec::with_capacity(frames as usize);
                for i in 0..frames as usize {
                    if silent || data.is_null() {
                        out.push([0.0, 0.0]);
                    } else {
                        let p = (data as *const f32).add(i * CAPTURE_CHANNELS as usize);
                        out.push([*p, *p.add(1)]);
                    }
                }
                if let Ok(mut s) = shared.lock() {
                    s.push(&out);
                }
                let _ = capture.ReleaseBuffer(frames);
            }
        }

        let _ = client.Stop();
        let _ = CloseHandle(buffer_event);
        let _ = CloseHandle(done_event);
    }
    Ok(())
}

/// A `PROPVARIANT` holding a blob, laid out by hand.
///
/// Filling in the `windows` type instead corrupts the heap: its `Drop` runs
/// `PropVariantClear`, which for `VT_BLOB` frees the blob pointer — and ours points
/// at a stack local, not at COM-allocated memory. This type has no destructor, and
/// `ActivateAudioInterfaceAsync` only reads through the pointer.
#[repr(C)]
struct BlobPropVariant {
    vt: u16,
    reserved: [u16; 3],
    cb_size: u32,
    pad: u32,
    p_blob_data: *mut u8,
}

const _: () = assert!(
    std::mem::size_of::<BlobPropVariant>() == std::mem::size_of::<PROPVARIANT>(),
    "BlobPropVariant must match the real PROPVARIANT layout"
);
