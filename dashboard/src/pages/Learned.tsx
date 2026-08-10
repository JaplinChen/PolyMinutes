import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowRight, Check, Pencil, Trash2 } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type KnownSpeaker, type LearnedCorrection } from '../services/app.api';
import { API_BASE_URL } from '../services/http';
import './Learned.css';

/**
 * What the room has picked up on its own: voices it can now name, and mistakes it will not repeat.
 *
 * Both are learned from ordinary use rather than configured — naming a speaker, correcting a line.
 * The reason this page exists is that learning silently is only acceptable if it can be inspected
 * and undone: a voiceprint attached to the wrong person, or a correction learned from a typo,
 * would otherwise keep applying with nowhere to see it.
 */
export function Learned() {
  const { t } = useTranslation();
  useDocumentTitle(t('learned.title'));
  const toast = useToast();

  const [tab, setTab] = useState<'voices' | 'corrections'>('voices');
  const [speakers, setSpeakers] = useState<KnownSpeaker[]>([]);
  const [corrections, setCorrections] = useState<LearnedCorrection[]>([]);
  const [langs, setLangs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [editingPair, setEditingPair] = useState<string | null>(null);
  const [pair, setPair] = useState({ wrong: '', right: '' });

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  useEffect(() => {
    let alive = true;
    Promise.all([appApi.knownSpeakers(), appApi.corrections(), appApi.getConfig()])
      .then(([voices, fixes, cfg]) => {
        if (!alive) return;
        setSpeakers(voices);
        setCorrections(fixes);
        setLangs(cfg.languages);
      })
      .catch(err => alive && fail(err))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const forgetSpeaker = async (name: string) => {
    setBusy(true);
    try {
      setSpeakers(await appApi.forgetSpeaker(name));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const setLanguage = async (name: string, language: string) => {
    setBusy(true);
    try {
      setSpeakers(await appApi.setSpeakerLanguage(name, language));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const renameSpeaker = async (name: string) => {
    const next = draft.trim();
    setEditing(null);
    if (!next || next === name) return;
    setBusy(true);
    try {
      setSpeakers(await appApi.renameSpeaker(name, next));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  // Both sides are editable: the recogniser's mishearing is worth fixing when it was itself a typo,
  // and the replacement is what actually lands in every future transcript.
  const saveCorrection = async (original: string) => {
    setEditingPair(null);
    if (pair.wrong.trim() === original && pair.right.trim() === corrections.find(c => c.wrong === original)?.right) return;
    setBusy(true);
    try {
      setCorrections(await appApi.editCorrection(original, { wrong: pair.wrong.trim(), right: pair.right.trim() }));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const forgetCorrection = async (wrong: string) => {
    setBusy(true);
    try {
      setCorrections(await appApi.forgetCorrection(wrong));
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
    <div className="etable-page learned-page">
      <PageHeader title={t('learned.title')} subtitle={t('learned.subtitle')} />

      <div className="learned-tabs" role="tablist">
        <button
          className={`learned-tab${tab === 'voices' ? ' active' : ''}`}
          role="tab"
          aria-selected={tab === 'voices'}
          onClick={() => setTab('voices')}
        >
          {t('learned.voices')}
          <span className="etable-count">{speakers.length}</span>
        </button>
        <button
          className={`learned-tab${tab === 'corrections' ? ' active' : ''}`}
          role="tab"
          aria-selected={tab === 'corrections'}
          onClick={() => setTab('corrections')}
        >
          {t('learned.corrections')}
          <span className="etable-count">{corrections.length}</span>
        </button>
      </div>

      {tab === 'voices' && (
      <section className="etable-panel">
        <p className="learned-note">{t('learned.voicesNote')}</p>
        {speakers.length === 0 ? (
          <p className="learned-empty">{t('learned.noVoices')}</p>
        ) : (
          <ul className="learned-list">
            {speakers.map(s => (
              <li key={s.name} className="learned-row">
                {editing === s.name ? (
                  <input
                    className="learned-rename"
                    autoFocus
                    value={draft}
                    onChange={e => setDraft(e.target.value)}
                    onBlur={() => renameSpeaker(s.name)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') e.currentTarget.blur();
                      if (e.key === 'Escape') setEditing(null);
                    }}
                  />
                ) : (
                  <button
                    className="learned-name"
                    title={t('learned.rename')}
                    onClick={() => {
                      setDraft(s.name);
                      setEditing(s.name);
                    }}
                  >
                    {s.name}
                  </button>
                )}
                <span className="learned-sessions">{t('learned.sessions', { count: s.sessions })}</span>
                <select
                  className="learned-lang"
                  value={s.language}
                  disabled={busy}
                  aria-label={t('learned.language')}
                  title={t('learned.language')}
                  onChange={e => setLanguage(s.name, e.target.value)}
                >
                  <option value="">{t('learned.langAuto')}</option>
                  {langs.map(c => (
                    <option key={c} value={c}>
                      {t(`lang.${c}`, { defaultValue: c })}
                    </option>
                  ))}
                </select>
                <audio
                  className="learned-clip"
                  controls
                  preload="none"
                  src={`${API_BASE_URL}/speakers/known/${encodeURIComponent(s.name)}/clip`}
                />
                <button
                  className="learned-forget"
                  disabled={busy}
                  title={t('learned.forget')}
                  onClick={() => forgetSpeaker(s.name)}
                >
                  <Trash2 size={16} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
      )}

      {tab === 'corrections' && (
      <section className="etable-panel">
        <p className="learned-note">{t('learned.correctionsNote')}</p>
        {corrections.length === 0 ? (
          <p className="learned-empty">{t('learned.noCorrections')}</p>
        ) : (
          <ul className="learned-list">
            {corrections.map(c => (
              <li key={c.wrong} className="learned-row">
                {editingPair === c.wrong ? (
                  // Enter saves from either box, Escape abandons — the same keys the speaker
                  // rename above already uses.
                  <form
                    className="learned-pair-edit"
                    onSubmit={e => {
                      e.preventDefault();
                      saveCorrection(c.wrong);
                    }}
                  >
                    <input
                      className="learned-rename"
                      autoFocus
                      aria-label={t('learned.editWrong')}
                      value={pair.wrong}
                      onChange={e => setPair(p => ({ ...p, wrong: e.target.value }))}
                      onKeyDown={e => e.key === 'Escape' && setEditingPair(null)}
                    />
                    <ArrowRight className="learned-arrow" size={14} />
                    <input
                      className="learned-rename"
                      aria-label={t('learned.editRight')}
                      value={pair.right}
                      onChange={e => setPair(p => ({ ...p, right: e.target.value }))}
                      onKeyDown={e => e.key === 'Escape' && setEditingPair(null)}
                    />
                    <button className="learned-save" disabled={busy} title={t('common.save')}>
                      <Check size={16} />
                    </button>
                  </form>
                ) : (
                  <>
                    <span className="learned-wrong">{c.wrong}</span>
                    <ArrowRight className="learned-arrow" size={14} />
                    <span className="learned-right">{c.right}</span>
                    <button
                      className="learned-edit"
                      disabled={busy}
                      title={t('learned.edit')}
                      onClick={() => {
                        setPair({ wrong: c.wrong, right: c.right });
                        setEditingPair(c.wrong);
                      }}
                    >
                      <Pencil size={16} />
                    </button>
                  </>
                )}
                <button
                  className="learned-forget"
                  disabled={busy}
                  title={t('learned.forget')}
                  onClick={() => forgetCorrection(c.wrong)}
                >
                  <Trash2 size={16} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
      )}
    </div>
  );
}
