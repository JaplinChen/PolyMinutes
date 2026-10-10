import { useState, useId } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Check, AlertTriangle, Plug, Plus, Trash2 } from 'lucide-react';
import { translateApi, type LlmProvider } from '../../services/api';
import { useToast } from '../Toast';
import { PROVIDERS, metaOf, apiKeyUrlForModel, type ProviderConfig } from './providerMeta';
import { LlmModelField } from './LlmModelField';
import { LlmApiKeyField } from './LlmApiKeyField';
import { isSubmitEnter } from '../../utils/ime';

interface Props {
  value: ProviderConfig;
  keySet: boolean;
  canWrite: boolean;
  allowNone: boolean; // fallback tabs may be left unset
  excludeProviders: LlmProvider[]; // providers chosen in other tabs — not re-selectable here
  onProviderChange: (p: LlmProvider | '') => void;
  onField: (patch: Partial<ProviderConfig>) => void;
}

export function ProviderConfigForm({ value, keySet, canWrite, allowNone, excludeProviders, onProviderChange, onField }: Props) {
  const { t } = useTranslation();
  const providerFieldId = useId();
  const endpointFieldId = useId();
  const styleFieldId = useId();
  const toast = useToast();

  const [models, setModels] = useState<string[]>([]);
  const [showKey, setShowKey] = useState(false);
  const [testing, setTesting] = useState(false);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [fbInput, setFbInput] = useState('');

  const addFallback = () => {
    const m = fbInput.trim();
    if (!m || value.model === m || value.fallbackModels.includes(m)) return;
    onField({ fallbackModels: [...value.fallbackModels, m] });
    setFbInput('');
  };
  const removeFallback = (m: string) => onField({ fallbackModels: value.fallbackModels.filter(x => x !== m) });

  const options = PROVIDERS.filter(p => p.value === value.provider || !excludeProviders.includes(p.value));

  if (!value.provider) {
    return (
      <div className="form-group">
        <label htmlFor={providerFieldId}>{t('llm.provider')}</label>
        <select id={providerFieldId} value="" disabled={!canWrite} onChange={e => onProviderChange(e.target.value as LlmProvider | '')}>
          <option value="">{t('llm.fallbackNone', { defaultValue: 'None (no fallback)' })}</option>
          {options.map(p => (
            <option key={p.value} value={p.value}>{p.meta.label}</option>
          ))}
        </select>
      </div>
    );
  }

  // An unknown provider — the backend's list ahead of this one, or a hand-edited llm.json — still
  // needs a usable form, so fall back to a generic shape: show the endpoint it has stored and offer
  // a key field, rather than hiding both and leaving the user unable to see their own config.
  const meta = metaOf(value.provider) ?? { label: value.provider, endpoint: '', showEndpoint: true, needsKey: true };
  const effectiveEndpoint = meta.showEndpoint ? value.endpoint || meta.endpoint : meta.endpoint;
  const probe = () => ({ provider: value.provider as LlmProvider, endpoint: effectiveEndpoint, model: value.model, apiKey: value.apiKey });

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      setTestResult(await translateApi.testLlm(probe()));
    } catch (err) {
      setTestResult({ ok: false, message: err instanceof Error ? err.message : 'unknown' });
    } finally {
      setTesting(false);
    }
  };

  const handleFetchModels = async () => {
    setFetchingModels(true);
    try {
      const { models: list } = await translateApi.listLlmModels(probe());
      setModels(list);
      if (list.length === 0) toast.error(t('llm.noModels'));
    } catch (err) {
      toast.error(t('llm.modelsFailed', { message: err instanceof Error ? err.message : 'unknown' }));
    } finally {
      setFetchingModels(false);
    }
  };

  const statusBadge = testResult && (
    <span className={`llm-status ${testResult.ok ? 'ok' : 'err'}`}>
      {testResult.ok ? <Check size={14} /> : <AlertTriangle size={14} />}
      {testResult.ok ? t('llm.testOk') : testResult.message}
    </span>
  );

  return (
    <>
      <div className="form-group">
        <label htmlFor={providerFieldId}>{t('llm.provider')}</label>
        <select id={providerFieldId} value={value.provider} disabled={!canWrite} onChange={e => onProviderChange(e.target.value as LlmProvider | '')}>
          {allowNone && <option value="">{t('llm.fallbackNone', { defaultValue: 'None (no fallback)' })}</option>}
          {options.map(p => (
            <option key={p.value} value={p.value}>{p.meta.label}</option>
          ))}
        </select>
      </div>

      {meta.showEndpoint && (
        <div className="form-group">
          <label htmlFor={endpointFieldId}>{t('llm.endpoint')}</label>
          <div className="llm-row">
            <input
              id={endpointFieldId}
              type="text"
              value={value.endpoint}
              disabled={!canWrite}
              placeholder={meta.endpoint}
              onChange={e => onField({ endpoint: e.target.value })}
            />
            <button className="btn-secondary" onClick={handleTest} disabled={testing || !value.endpoint}>
              {testing ? <Loader2 size={16} className="animate-spin" /> : <Plug size={16} />}
              {t('llm.test')}
            </button>
          </div>
          {statusBadge}
        </div>
      )}

      <LlmModelField
        model={value.model}
        models={models}
        canWrite={canWrite}
        onChange={v => onField({ model: v })}
        fetchingModels={fetchingModels}
        fetchDisabled={!value.endpoint && !meta.endpoint}
        onFetchModels={() => void handleFetchModels()}
      />

      <div className="form-group">
        <label>{t('llm.fallbackModels', { defaultValue: 'Fallback models' })}</label>
        <span className="llm-hint">
          {t('llm.perProviderFallbackHint', { defaultValue: 'Extra models of this provider, tried in order after the model above.' })}
        </span>
        {canWrite && (
          <div className="llm-row">
            <input
              type="text"
              value={fbInput}
              placeholder={t('llm.fallbackPlaceholder', { defaultValue: 'Model name' })}
              onChange={e => setFbInput(e.target.value)}
              onKeyDown={e => isSubmitEnter(e) && (e.preventDefault(), addFallback())}
            />
            <button className="btn-secondary" onClick={addFallback} disabled={!fbInput.trim()}>
              <Plus size={16} />
              {t('llm.add', { defaultValue: 'Add' })}
            </button>
          </div>
        )}
        <ul className="llm-fallback-list">
          {value.fallbackModels.map(m => (
            <li key={m}>
              <span>{m}</span>
              {canWrite && (
                <button
                  className="llm-fallback-del"
                  onClick={() => removeFallback(m)}
                  title={t('llm.removeFallback')}
                  aria-label={`${t('llm.removeFallback')}: ${m}`}
                >
                  <Trash2 size={14} />
                </button>
              )}
            </li>
          ))}
        </ul>
      </div>

      {meta.needsKey && (
        <LlmApiKeyField
          meta={meta}
          apiKeyUrl={apiKeyUrlForModel(meta, value.model)}
          apiKey={value.apiKey}
          keySet={keySet}
          canWrite={canWrite}
          showKey={showKey}
          toggleShowKey={() => setShowKey(v => !v)}
          onChange={v => onField({ apiKey: v })}
          testing={testing}
          onTest={handleTest}
          statusBadge={statusBadge}
        />
      )}

      <div className="form-group">
        <label htmlFor={styleFieldId}>{t('llm.style')}</label>
        <div className="llm-slider-row">
          <span>{t('llm.stylePrecise')}</span>
          <input
            id={styleFieldId}
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={value.temperature}
            disabled={!canWrite}
            onChange={e => onField({ temperature: Number(e.target.value) })}
          />
          <span>{t('llm.styleFluent')}</span>
          <span className="llm-slider-val">{value.temperature.toFixed(1)}</span>
        </div>
      </div>
    </>
  );
}
