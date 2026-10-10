import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { KeyRound, Trash2, Plus, ExternalLink } from 'lucide-react';
import { keyProxyApi, type KeyStatus } from '../services/api';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { useResizableCol } from '../hooks/useResizableCol';
import { useRole } from '../hooks/useRole';
import { NO_AUTOFILL } from '../utils/noAutofill';
import { isSubmitEnter } from '../utils/ime';
import { useToast } from '../components/Toast';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import './KeyProxy.css';

/**
 * Providers report what is LEFT, not what was spent, and the window resets on its own schedule — so
 * "used" is derived per reading rather than accumulated. Audio seconds are surfaced too because for
 * transcription they run out first: 7200s at ~5s per voice note is ~1440 notes against a 2000 request
 * ceiling, so the request count alone would suggest more headroom than actually exists.
 */
const quotaParts = (q: NonNullable<KeyStatus['quota']>) => ({
  usedRequests: q.limitRequests - q.remainingRequests,
  limitRequests: q.limitRequests,
  usedAudio: Math.round(q.limitAudioSeconds - q.remainingAudioSeconds),
  limitAudio: q.limitAudioSeconds,
});

// Providers the llm-key-proxy supports that make sense for free-tier rotation here.
const PROVIDERS = ['gemini', 'groq', 'openai', 'anthropic', 'mistral', 'openrouter', 'nvidia_nim'];

const APPLY_URLS: Record<string, string> = {
  gemini: 'https://aistudio.google.com/apikey',
  groq: 'https://console.groq.com/keys',
  openai: 'https://platform.openai.com/api-keys',
  anthropic: 'https://console.anthropic.com/settings/keys',
  mistral: 'https://console.mistral.ai/api-keys',
  openrouter: 'https://openrouter.ai/keys',
  nvidia_nim: 'https://build.nvidia.com/settings/api-keys',
};

// Module level, not nested in KeyProxy: a component created during render is a new type every render,
// which remounts the subtree instead of updating it (react-hooks/static-components).
function RH({ col, onStart }: { col: string; onStart: (col: string) => (e: React.MouseEvent) => void }) {
  return <span className="kp-resize-handle" aria-hidden="true" onMouseDown={onStart(col)} />;
}

export function KeyProxy() {
  const { t } = useTranslation();
  useDocumentTitle(t('keyproxy.title'));
  const { canWrite } = useRole();
  const toast = useToast();
  const { ref: tableRef, startResize } = useResizableCol('keyproxy-cols');

  const [keys, setKeys] = useState<KeyStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState(PROVIDERS[0]);
  const [apiKey, setApiKey] = useState('');
  const [account, setAccount] = useState('');

  const fail = (err: unknown) =>
    toast.error(t('common.failed', { message: err instanceof Error ? err.message : 'unknown' }));

  useEffect(() => {
    let active = true;
    keyProxyApi
      .list()
      .then(list => active && setKeys(list))
      .catch(err => active && fail(err))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  const add = async () => {
    if (!apiKey.trim()) return;
    setBusy(true);
    try {
      setKeys(await keyProxyApi.add(provider, apiKey.trim(), account.trim()));
      setApiKey('');
      setAccount('');
      toast.success(t('keyproxy.added'));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (k: KeyStatus) => {
    setBusy(true);
    try {
      setKeys(await keyProxyApi.remove(k.provider, k.index));
      toast.success(t('keyproxy.removed'));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  if (loading) {
    return <PageSkeleton rows={5} />;
  }

  return (
    <div className="etable-page keyproxy-page">
      <PageHeader title={t('keyproxy.title')} subtitle={t('keyproxy.subtitle')} />

      <p className="keyproxy-note">{t('keyproxy.accountNote')}</p>

      {canWrite && (
        <div className="keyproxy-addbar">
          <select
            className="keyproxy-select"
            value={provider}
            onChange={e => setProvider(e.target.value)}
            disabled={busy}
            aria-label={t('keyproxy.provider')}
          >
            {PROVIDERS.map(p => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          {APPLY_URLS[provider] && (
            <a className="keyproxy-apply" href={APPLY_URLS[provider]} target="_blank" rel="noreferrer">
              <ExternalLink size={14} />
              {t('llm.apiKeyApply')}
            </a>
          )}
          <input
            className="keyproxy-input"
            type="password"
            aria-label={t('keyproxy.key')}
            {...NO_AUTOFILL}
            placeholder={t('keyproxy.keyPlaceholder')}
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            onKeyDown={e => isSubmitEnter(e) && !busy && add()}
            disabled={busy}
          />
          <input
            className="keyproxy-account-input"
            type="text"
            aria-label={t('keyproxy.account')}
            autoComplete="off"
            placeholder={t('keyproxy.accountPlaceholder')}
            value={account}
            onChange={e => setAccount(e.target.value)}
            onKeyDown={e => isSubmitEnter(e) && !busy && add()}
            disabled={busy}
          />
          <button className="btn-primary" onClick={add} disabled={busy || !apiKey.trim()}>
            <Plus size={16} />
            {t('keyproxy.add')}
          </button>
        </div>
      )}

      <div className="etable-panel">
        <div className="etable-panel-title">{t('keyproxy.keys')}</div>
        {keys.length === 0 ? (
          <div className="keyproxy-empty">
            <KeyRound size={32} strokeWidth={1} />
            <span>{t('keyproxy.empty')}</span>
          </div>
        ) : (
          <div className="keyproxy-table-scroll">
          <table className="keyproxy-table" ref={tableRef}>
            <colgroup>
              <col style={{ width: 'var(--col-provider, 8rem)' }} />
              <col style={{ width: 'var(--col-account, 9rem)' }} />
              <col style={{ width: 'var(--col-keycol, auto)' }} />
              <col style={{ width: 'var(--col-status, 7rem)' }} />
              <col style={{ width: 'var(--col-requests, 6rem)' }} />
              {/* Wider than the neighbours: holds "1234/2000", not a single number. */}
              <col style={{ width: 'var(--col-quota, 8rem)' }} />
              <col style={{ width: 'var(--col-failures, 6rem)' }} />
              {canWrite && <col style={{ width: '3rem' }} />}
            </colgroup>
            <thead>
              <tr>
                <th data-col="provider">{t('keyproxy.provider')}<RH col="provider" onStart={startResize} /></th>
                <th data-col="account">{t('keyproxy.account')}<RH col="account" onStart={startResize} /></th>
                <th data-col="keycol">{t('keyproxy.key')}<RH col="keycol" onStart={startResize} /></th>
                <th data-col="status">{t('keyproxy.status')}<RH col="status" onStart={startResize} /></th>
                <th className="keyproxy-num" data-col="requests">{t('keyproxy.requests')}<RH col="requests" onStart={startResize} /></th>
                <th className="keyproxy-num" data-col="quota">{t('keyproxy.quota')}<RH col="quota" onStart={startResize} /></th>
                <th className="keyproxy-num" data-col="failures">{t('keyproxy.failures')}<RH col="failures" onStart={startResize} /></th>
                {canWrite && <th />}
              </tr>
            </thead>
            <tbody>
              {keys.map(k => (
                <tr key={`${k.provider}-${k.index}`}>
                  <td>{k.provider}</td>
                  <td>{k.account || <span className="keyproxy-muted">—</span>}</td>
                  <td className="keyproxy-mono">{k.masked}</td>
                  <td>
                    <span className={`keyproxy-badge keyproxy-badge-${k.status}`}>{k.status}</span>
                  </td>
                  <td className="keyproxy-num">
                    {k.requestCount}
                    {k.voiceRequestCount > 0 && (
                      <span className="keyproxy-muted" title={t('keyproxy.voiceHint')}>
                        {' '}
                        ({t('keyproxy.voiceCount', { count: k.voiceRequestCount })})
                      </span>
                    )}
                  </td>
                  <td className="keyproxy-num">
                    {k.quota ? (
                      <span title={t('keyproxy.quotaHint', { ...quotaParts(k.quota) })}>
                        {quotaParts(k.quota).usedRequests}/{k.quota.limitRequests}
                      </span>
                    ) : (
                      <span className="keyproxy-muted" title={t('keyproxy.quotaNoneHint')}>
                        —
                      </span>
                    )}
                  </td>
                  <td className="keyproxy-num">{k.failureCount}</td>
                  {canWrite && (
                    <td>
                      <button
                        className="keyproxy-del"
                        onClick={() => remove(k)}
                        disabled={busy}
                        aria-label={t('keyproxy.remove')}
                        title={t('keyproxy.remove')}
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default KeyProxy;
