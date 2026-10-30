import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, Loader2, Mic, Square } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type AudioDevice, type RecordingStatus } from '../services/app.api';
import './Capture.css';

const POLL_MS = 500;
// Peak stays below this for the whole window and we call it silent. Chosen well under speech
// level but above the dither floor of an idle virtual cable.
const SILENT_PEAK = 0.002;
const SILENT_AFTER_SECONDS = 5;

export function Capture() {
  const { t } = useTranslation();
  useDocumentTitle(t('capture.title'));
  const toast = useToast();

  const [status, setStatus] = useState<RecordingStatus | null>(null);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [outputs, setOutputs] = useState<string[]>([]);
  const [configured, setConfigured] = useState('');
  const [loopback, setLoopback] = useState('');
  const [deviceError, setDeviceError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loudestSeen, setLoudestSeen] = useState(0);
  // null until the config answers, so the warning cannot flash on before anyone knows it is true.
  const [translatorReady, setTranslatorReady] = useState<boolean | null>(null);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  const loadDevices = () =>
    appApi
      .devices()
      .then(d => {
        setDevices(d.devices);
        setOutputs(d.outputs ?? []);
        setConfigured(d.configured);
        setDeviceError(d.error);
      })
      .catch(fail);

  useEffect(() => {
    loadDevices();
    // The server computes this on every config read and the page never asked: with no key set, a
    // recording is transcribed but never translated, and the only mention was a server-side log
    // line. A supported mode — so a warning with the way to fix it, not a refusal.
    appApi
      .getConfig()
      .then(c => {
        setTranslatorReady(c.translatorReady);
        setLoopback(c.loopbackDevice);
      }).catch(() => undefined);
    const tick = () =>
      appApi
        .recordingStatus()
        .then(s => {
          setStatus(s);
          setLoudestSeen(prev => (s.recording ? Math.max(prev, s.peak) : 0));
        })
        .catch(() => undefined);
    tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => window.clearInterval(id);
  }, []);

  const selectDevice = async (patch: { inputDevice?: string; loopbackDevice?: string }) => {
    setBusy(true);
    try {
      const c = await appApi.putConfig(patch);
      setLoopback(c.loopbackDevice);
      await loadDevices();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const toggle = async () => {
    setBusy(true);
    try {
      setLoudestSeen(0);
      setStatus(status?.recording ? await appApi.stopRecording() : await appApi.startRecording());
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const recording = status?.recording ?? false;
  // A saved speaker that is no longer plugged in would otherwise render as "none" and then fail on start.
  const loopbackMissing =
    !!loopback &&
    !outputs.some(name => `system-output:${name}` === loopback) &&
    !devices.some(d => d.name === loopback);
  // The failure this catches: "no output" achieved by muting the playback device, which makes
  // loopback capture silent. Without this the meeting looks fine until the transcript is empty.
  const silent = recording && status!.seconds > SILENT_AFTER_SECONDS && loudestSeen < SILENT_PEAK;

  return (
    <div className="etable-page capture-page">
      <PageHeader
        title={t('capture.title')}
        subtitle={t('capture.subtitle')}
        actions={
          <button className={recording ? 'btn-danger' : 'btn-primary'} onClick={toggle} disabled={busy}>
            {busy ? <Loader2 size={18} className="animate-spin" /> : recording ? <Square size={18} /> : <Mic size={18} />}
            {recording ? t('capture.stop') : t('capture.start')}
          </button>
        }
      />

      {deviceError && (
        <div className="capture-alert" role="alert">
          <AlertTriangle size={18} />
          <span>{deviceError}</span>
        </div>
      )}

      {translatorReady === false && (
        <div className="capture-alert" role="alert">
          <AlertTriangle size={18} />
          <span>
            {t('capture.noTranslator')} <Link to="/settings/llm">{t('capture.noTranslatorLink')}</Link>
          </span>
        </div>
      )}

      {silent && (
        <div className="capture-alert" role="alert">
          <AlertTriangle size={18} />
          <span>{t('capture.silentWarning')}</span>
        </div>
      )}

      {recording && status!.loopbackError && (
        <div className="capture-alert" role="alert">
          <AlertTriangle size={18} />
          <span>{t('capture.loopbackFailed', { error: status!.loopbackError })}</span>
        </div>
      )}

      <section className="etable-panel">
        <h3 className="etable-panel-title">{t('capture.device')}</h3>
        <p className="capture-hint">{t('capture.deviceHint')}</p>
        <select
          className="capture-select"
          aria-label={t('capture.device')}
          value={configured}
          onChange={e => selectDevice({ inputDevice: e.target.value })}
          disabled={busy || recording}
        >
          <option value="">{t('capture.systemDefault')}</option>
          {devices.map(d => (
            <option key={`${d.index}-${d.hostapi}`} value={d.name}>
              {d.name} — {d.hostapi}
            </option>
          ))}
        </select>
      </section>

      <section className="etable-panel">
        <h3 className="etable-panel-title">{t('capture.loopback')}</h3>
        <p className="capture-hint">{t('capture.loopbackHint')}</p>
        <select
          className="capture-select"
          aria-label={t('capture.loopback')}
          value={loopback}
          onChange={e => selectDevice({ loopbackDevice: e.target.value })}
          disabled={busy || recording}
        >
          <option value="">{t('capture.loopbackNone')}</option>
          {loopbackMissing && <option value={loopback}>{t('capture.loopbackMissing', { name: loopback.replace(/^system-output:/, '') })}</option>}
          {outputs.map(name => (
            <option key={name} value={`system-output:${name}`}>
              {t('capture.loopbackSystem')} — {name}
            </option>
          ))}
          {devices.map(d => (
            <option key={`${d.index}-${d.hostapi}`} value={d.name}>
              {d.name} — {d.hostapi}
            </option>
          ))}
        </select>
      </section>

      <section className="etable-panel">
        <h3 className="etable-panel-title">{t('capture.level')}</h3>
        <div className="capture-meter" aria-hidden="true">
          <div className="capture-meter-fill" style={{ width: `${Math.min(100, (status?.peak ?? 0) * 300)}%` }} />
        </div>
        <dl className="capture-stats">
          <div>
            <dt>{t('capture.elapsed')}</dt>
            <dd>{(status?.seconds ?? 0).toFixed(1)}s</dd>
          </div>
          <div>
            <dt>{t('capture.peak')}</dt>
            <dd>{(status?.peak ?? 0).toFixed(4)}</dd>
          </div>
          <div>
            <dt>{t('capture.backlog')}</dt>
            <dd className={status && status.backlog > 100 ? 'capture-bad' : undefined}>{status?.backlog ?? 0}</dd>
          </div>
          <div>
            <dt>{t('capture.dropped')}</dt>
            <dd className={status && status.droppedBlocks > 0 ? 'capture-bad' : undefined}>
              {status?.droppedBlocks ?? 0}
            </dd>
          </div>
          <div>
            <dt>{t('capture.errors')}</dt>
            <dd className={status && status.errors > 0 ? 'capture-bad' : undefined}>{status?.errors ?? 0}</dd>
          </div>
        </dl>
        {status?.path && <p className="capture-path">{status.path}</p>}
      </section>
    </div>
  );
}

export default Capture;
