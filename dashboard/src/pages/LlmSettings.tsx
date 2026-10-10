import { useState, useEffect, useEffectEvent, useId } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Save } from 'lucide-react';
import { translateApi, type LlmProvider, type LlmProviderSaved } from '../services/api';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { useRole } from '../hooks/useRole';
import { PageHeader } from '../components/PageHeader';
import { useToast } from '../components/Toast';
import { metaOf, parseFallbackEntry, emptyProviderConfig, type ProviderConfig } from '../components/llm/providerMeta';
import { ProviderConfigForm } from '../components/llm/ProviderConfigForm';
import './Translate.css';

const TAB_KEYS = ['tabPrimary', 'tabFallback1', 'tabFallback2'] as const;

export function LlmSettings() {
  const { t } = useTranslation();
  useDocumentTitle(t('nav.llm', { defaultValue: 'LLM Settings' }));
  const { canWrite } = useRole();
  const toast = useToast();

  // Index 0 = primary, 1 = fallback 1, 2 = fallback 2. Each holds a full provider config.
  const [tabs, setTabs] = useState<ProviderConfig[]>([emptyProviderConfig(), emptyProviderConfig(), emptyProviderConfig()]);
  const [pcfgs, setPcfgs] = useState<Record<string, LlmProviderSaved>>({});
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  // Per-function model overrides; empty means "use the primary model". Global, not per-tab.
  const [translateModel, setTranslateModel] = useState('');
  const [summaryModel, setSummaryModel] = useState('');
  const translateId = useId();
  const summaryId = useId();
  const modelListId = useId();

  const keySet = (p: LlmProvider | ''): boolean => (p ? Boolean(pcfgs[p]?.apiKeySet) : false);

  const mapFromConfig = (c: Awaited<ReturnType<typeof translateApi.getConfig>>): void => {
    // Fold the active provider's apiKeySet into pcfgs so keySet() is uniform across tabs.
    const merged = { ...c.llmProviderConfigs, [c.llmProvider]: { ...c.llmProviderConfigs?.[c.llmProvider], apiKeySet: c.apiKeySet } };
    setPcfgs(merged);
    // Group the flat "provider:model" chain by provider: the primary provider's entries become the
    // primary tab's fallbackModels; each other provider (in first-seen order) becomes a fallback tab.
    const parsed = (c.llmFallbackModels ?? []).map(e => parseFallbackEntry(e, c.llmProvider));
    const order: LlmProvider[] = [];
    const byProvider: Record<string, string[]> = {};
    for (const p of parsed) {
      if (p.provider === c.llmProvider) continue;
      if (!byProvider[p.provider]) { byProvider[p.provider] = []; order.push(p.provider); }
      byProvider[p.provider].push(p.model);
    }
    const primary: ProviderConfig = {
      provider: c.llmProvider,
      endpoint: c.llmEndpoint,
      model: c.llmModel,
      apiKey: '', // masked; blank means "keep stored key"
      temperature: c.llmTemperature,
      fallbackModels: parsed.filter(p => p.provider === c.llmProvider).map(p => p.model),
    };
    const fbTab = (prov?: LlmProvider): ProviderConfig => {
      if (!prov) return emptyProviderConfig();
      const ms = byProvider[prov];
      const saved = merged[prov];
      return {
        provider: prov,
        endpoint: saved?.endpoint ?? metaOf(prov)?.endpoint ?? '',
        model: ms[0],
        apiKey: '',
        temperature: saved?.temperature ?? 0,
        fallbackModels: ms.slice(1),
      };
    };
    setTabs([primary, fbTab(order[0]), fbTab(order[1])]);
    setTranslateModel(c.llmTranslateModel ?? '');
    setSummaryModel(c.llmSummaryModel ?? '');
  };

  // A UI-language switch gives ToastProvider a new `t` and so a new `toast`; reading both through an
  // effect event keeps the load once per mount, so switching language never wipes unsaved edits.
  const loadFailed = useEffectEvent((err: unknown) =>
    toast.error(t('llm.loadFailed', { message: err instanceof Error ? err.message : 'unknown' })));
  useEffect(() => {
    let alive = true;
    translateApi
      .getConfig()
      .then(c => {
        if (!alive) return;
        mapFromConfig(c);
        setLoaded(true);
      })
      .catch(err => alive && loadFailed(err))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const patchTab = (idx: number, patch: Partial<ProviderConfig>) =>
    setTabs(prev => prev.map((tab, i) => (i === idx ? { ...tab, ...patch } : tab)));

  // Switching a tab's provider restores that provider's saved settings (or its defaults on first use).
  const changeProvider = (idx: number, p: LlmProvider | '') => {
    if (!p) return setTabs(prev => prev.map((tab, i) => (i === idx ? emptyProviderConfig() : tab)));
    const saved = pcfgs[p];
    patchTab(idx, {
      provider: p,
      endpoint: saved?.endpoint ?? metaOf(p)?.endpoint ?? '',
      model: saved?.model ?? '',
      apiKey: '',
      temperature: saved?.temperature ?? 0,
      fallbackModels: [],
    });
  };

  // Providers chosen in the OTHER tabs — not re-selectable here (one provider per tab).
  const excludeFor = (idx: number): LlmProvider[] =>
    tabs.filter((_, i) => i !== idx).map(tab => tab.provider).filter((p): p is LlmProvider => p !== '');

  const effectiveEndpoint = (tab: ProviderConfig): string => {
    const meta = metaOf(tab.provider);
    if (!meta) return tab.endpoint;
    return meta.showEndpoint ? tab.endpoint || meta.endpoint : meta.endpoint;
  };

  const handleSave = async () => {
    const primary = tabs[0];
    if (!primary.provider || !primary.model.trim()) {
      return toast.error(t('llm.primaryRequired', { defaultValue: 'The primary tab needs a provider and model.' }));
    }
    setSaving(true);
    try {
      const fbs = [tabs[1], tabs[2]].filter(tab => tab.provider && tab.model.trim());
      // Fold every set tab's config into providerConfigs so cross-provider fallback can resolve each key.
      const llmProviderConfigs: Record<string, LlmProviderSaved> = { ...pcfgs };
      for (const tab of [primary, ...fbs]) {
        llmProviderConfigs[tab.provider] = {
          endpoint: effectiveEndpoint(tab),
          model: tab.model,
          apiKey: tab.apiKey, // '' round-trips as "keep stored key"
          temperature: tab.temperature,
        };
      }
      // Flatten to an ordered "provider:model" chain: primary's extra models, then each fallback tab's
      // model followed by its own extra models. The primary tab's own `model` is the active llmModel.
      const clean = (arr: string[]) => arr.map(s => s.trim()).filter(Boolean);
      const chain = [
        ...clean(primary.fallbackModels).map(m => `${primary.provider}:${m}`),
        ...fbs.flatMap(tab => [`${tab.provider}:${tab.model.trim()}`, ...clean(tab.fallbackModels).map(m => `${tab.provider}:${m}`)]),
      ];
      const saved = await translateApi.updateConfig({
        llmProvider: primary.provider,
        llmEndpoint: effectiveEndpoint(primary),
        llmModel: primary.model,
        llmTranslateModel: translateModel.trim(),
        llmSummaryModel: summaryModel.trim(),
        llmApiKey: primary.apiKey,
        llmTemperature: primary.temperature,
        llmFallbackModels: chain,
        llmProviderConfigs,
      });
      mapFromConfig(saved);
      toast.success(t('llm.saved'));
    } catch (err) {
      toast.error(t('llm.saveFailed', { message: err instanceof Error ? err.message : 'unknown' }));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="translate-page translate-loading">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  return (
    <div className="translate-page">
      <PageHeader
        title={t('nav.llm', { defaultValue: 'LLM Settings' })}
        subtitle={t('llm.subtitle')}
        actions={
          canWrite && (
            <button className="btn-primary" onClick={handleSave} disabled={saving || !loaded}>
              {saving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
              {t('llm.save')}
            </button>
          )
        }
      />

      <div className="translate-content translate-content--single">
        <section className="translate-panel">
          <div className="llm-tabs" role="tablist">
            {tabs.map((tab, i) => (
              <button
                key={TAB_KEYS[i]}
                role="tab"
                aria-selected={active === i}
                className={`llm-tab ${active === i ? 'active' : ''}`}
                onClick={() => setActive(i)}
              >
                <span className="llm-tab-name">{t(`llm.${TAB_KEYS[i]}`, { defaultValue: TAB_KEYS[i] })}</span>
                <span className="llm-tab-sub">{tab.provider ? (metaOf(tab.provider)?.label ?? tab.provider) : t('llm.fallbackNone', { defaultValue: '—' })}</span>
              </button>
            ))}
          </div>

          <ProviderConfigForm
            key={`${active}-${tabs[active].provider}`}
            value={tabs[active]}
            keySet={keySet(tabs[active].provider)}
            canWrite={canWrite}
            allowNone={active > 0}
            excludeProviders={excludeFor(active)}
            onProviderChange={p => changeProvider(active, p)}
            onField={patch => patchTab(active, patch)}
          />
        </section>

        <section className="translate-panel">
          <p className="llm-hint">{t('llm.perFunctionHint')}</p>
          <div className="form-group">
            <label htmlFor={translateId}>{t('llm.translateModel')}</label>
            <input
              id={translateId}
              type="text"
              list={modelListId}
              value={translateModel}
              disabled={!canWrite}
              placeholder={t('llm.perFunctionPlaceholder')}
              onChange={e => setTranslateModel(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label htmlFor={summaryId}>{t('llm.summaryModel')}</label>
            <input
              id={summaryId}
              type="text"
              list={modelListId}
              value={summaryModel}
              disabled={!canWrite}
              placeholder={t('llm.perFunctionPlaceholder')}
              onChange={e => setSummaryModel(e.target.value)}
            />
          </div>
          <datalist id={modelListId}>
            {[...new Set([tabs[0].model, ...tabs[0].fallbackModels, tabs[1].model, tabs[2].model].filter(Boolean))].map(m => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </section>
      </div>
    </div>
  );
}

export default LlmSettings;
