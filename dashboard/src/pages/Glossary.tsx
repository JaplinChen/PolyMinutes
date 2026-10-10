import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { BookMarked, Plus, Trash2 } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type GlossaryTerm } from '../services/app.api';
import { isSubmitEnter } from '../utils/ime';
import './Glossary.css';

const MODES = ['translate', 'keep', 'hint', 'protect'] as const;

const emptyDraft = () => ({
  source: '',
  mode: 'translate' as GlossaryTerm['mode'],
  category: '',
  targets: {} as Record<string, string>,
});

export function Glossary() {
  const { t } = useTranslation();
  useDocumentTitle(t('glossary.title'));
  const toast = useToast();

  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [languages, setLanguages] = useState<string[]>([]);
  const [draft, setDraft] = useState(emptyDraft());
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [clash, setClash] = useState<{ text: string; count: number }[]>([]);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  useEffect(() => {
    let alive = true;
    Promise.all([appApi.glossary(), appApi.getConfig()])
      .then(([list, cfg]) => {
        if (!alive) return;
        setTerms(list);
        setLanguages(cfg.languages);
      })
      .catch(err => alive && fail(err))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const add = async () => {
    if (!draft.source.trim()) return;

    // The first press reports what this term would overwrite; the second goes ahead. Nothing is
    // blocked — a homophone may be exactly what you meant — but it is never silent.
    if (!clash.length) {
      const found = await collisionsFor(draft.source);
      if (found.length) return setClash(found);
    }

    setBusy(true);
    try {
      setTerms(await appApi.addTerm(draft));
      setDraft(emptyDraft());
      setClash([]);
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  // The corrector rewrites anything whose pinyin matches a term, and Mandarin supplies
  // homophones for almost everything, so adding one is not obviously destructive: 料號 and 料耗
  // are both liaohao, 料耗 is a term of the trade, and adding 料號 rewrote it forty-two times
  // without saying so.
  //
  // Checked on the way in rather than as the field loses focus. Two interactions in this app have
  // now depended on a focus event and not received one, and a warning that sometimes fires is
  // worse than none. Asked against the meetings already recorded, so the answer is about what
  // these people say rather than what Mandarin permits.
  const collisionsFor = async (source: string) => {
    try {
      return (await appApi.termCollisions(source.trim())).collisions;
    } catch {
      return [];
    }
  };

  const remove = async (term: GlossaryTerm) => {
    setBusy(true);
    try {
      setTerms(await appApi.removeTerm(term.source, term.lang));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  if (loading) {
    return <PageSkeleton />;
  }

  return (
    <div className="etable-page gloss-page">
      <PageHeader title={t('glossary.title')} subtitle={t('glossary.subtitle')} />

      <section className="etable-panel">
        {clash.length > 0 && (
          <p className="gloss-clash">
            {t('glossary.clashWarning', {
              list: clash.map(c => `${c.text} (${c.count})`).join('、'),
            })}
          </p>
        )}
        <div className="gloss-add">
          <label className="gloss-field gloss-field-source">
            <span>{t('glossary.source')}</span>
            <input
              className="gloss-input"
              value={draft.source}
              onChange={e => { setDraft({ ...draft, source: e.target.value }); setClash([]); }}
              onKeyDown={e => isSubmitEnter(e) && !busy && add()}
            />
          </label>
          <label className="gloss-field gloss-field-mode">
            <span>{t('glossary.modeLabel')}</span>
            <select
              value={draft.mode}
              onChange={e => setDraft({ ...draft, mode: e.target.value as GlossaryTerm['mode'] })}
            >
              {MODES.map(m => (
                <option key={m} value={m}>
                  {t(`glossary.mode.${m}`)}
                </option>
              ))}
            </select>
          </label>
          {/* What the chosen mode does, rather than all four at once. `protect` had an explanation
              written for it that nothing ever rendered. */}
          <p className="gloss-hint gloss-mode-help">{t(`glossary.modeHelp.${draft.mode}`)}</p>
          {/* Only `translate` needs per-language wording — `keep` and `hint` have no target text. */}
          {draft.mode === 'translate' && (
            <div className="gloss-targets">
              {languages.map(lang => (
                <label key={lang} className="gloss-field">
                  <span>{t(`lang.${lang}`, { defaultValue: lang })}</span>
                  <input
                    className="gloss-input gloss-target"
                    value={draft.targets[lang] ?? ''}
                    onChange={e => setDraft({ ...draft, targets: { ...draft.targets, [lang]: e.target.value } })}
                  />
                </label>
              ))}
            </div>
          )}
          <label className="gloss-field gloss-field-category">
            <span>{t('glossary.category')}</span>
            <input
              className="gloss-input gloss-category"
              placeholder={t('glossary.categoryPlaceholder')}
              value={draft.category}
              onChange={e => setDraft({ ...draft, category: e.target.value })}
            />
          </label>
          <button className="btn-primary gloss-submit" onClick={add} disabled={busy || !draft.source.trim()}>
            <Plus size={16} />
            {t('glossary.add')}
          </button>
        </div>
      </section>

      <section className="etable-panel">
        <div className="etable-panel-title">
          {t('glossary.terms')}
          <span className="etable-count">{terms.length}</span>
        </div>
        {terms.length === 0 ? (
          <div className="gloss-empty">
            <BookMarked size={32} strokeWidth={1} />
            <span>{t('glossary.empty')}</span>
          </div>
        ) : (
          <div className="gloss-scroll">
            <table className="gloss-table">
              <thead>
                <tr>
                  <th>{t('glossary.source')}</th>
                  <th>{t('glossary.modeLabel')}</th>
                  {languages.map(l => (
                    <th key={l}>{t(`lang.${l}`, { defaultValue: l })}</th>
                  ))}
                  <th>{t('glossary.category')}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {terms.map(term => (
                  <tr key={`${term.source}/${term.lang}`}>
                    <td className="gloss-src">{term.source}</td>
                    <td>
                      <span className={`gloss-badge gloss-badge-${term.mode}`}>{t(`glossary.mode.${term.mode}`)}</span>
                    </td>
                    {languages.map(l => (
                      <td key={l}>{term.mode === 'translate' ? (term.targets[l] ?? '—') : '—'}</td>
                    ))}
                    <td className="gloss-muted">{term.category || '—'}</td>
                    <td>
                      <button
                        className="gloss-del"
                        onClick={() => remove(term)}
                        disabled={busy}
                        title={t('glossary.remove')}
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default Glossary;
