import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowRight, Check, Pencil, Trash2, X } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type KnownSpeaker, type LearnedCorrection, type SpeakerClip } from '../services/app.api';
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
  // The name (or correction key) of the row whose request is in flight; null when idle. Every
  // control still locks while any request runs — the value only says where to show the feedback.
  const [busy, setBusy] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [editingPair, setEditingPair] = useState<string | null>(null);
  const [pair, setPair] = useState({ wrong: '', right: '' });
  const [search, setSearch] = useState('');

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  useEffect(() => {
    let alive = true;
    const load = (initial: boolean) => {
      Promise.all([appApi.knownSpeakers(), appApi.corrections(), appApi.getConfig()])
        .then(([voices, fixes, cfg]) => {
          if (!alive) return;
          setSpeakers(voices);
          setCorrections(fixes);
          setLangs(cfg.languages);
        })
        .catch(err => alive && fail(err))
        .finally(() => initial && alive && setLoading(false));
    };
    load(true);
    // Names and voiceprints change from the transcript pages too; refetch on return so the page
    // never shows a speaker deleted or renamed elsewhere.
    const onVisible = () => {
      if (document.visibilityState === 'visible') load(false);
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      alive = false;
      document.removeEventListener('visibilitychange', onVisible);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The server orders speakers by meeting count, and deleting or reassigning a sample changes
  // the counts — re-sorting mid-edit would yank the row out from under the cursor. Responses
  // keep the order already on screen; names not shown yet append at the end. A rename passes
  // the {from, to} mapping so the renamed row keeps the old name's position.
  const stableOrder = (prev: KnownSpeaker[], next: KnownSpeaker[], renamed?: { from: string; to: string }) => {
    const pos = new Map(prev.map((s, i) => [s.name, i]));
    const at = (name: string) =>
      pos.get(name) ?? (renamed && name === renamed.to ? pos.get(renamed.from) : undefined) ?? prev.length;
    return [...next].sort((a, b) => at(a.name) - at(b.name));
  };

  const forgetSpeaker = async (name: string) => {
    if (!window.confirm(t('learned.confirmForget', { name }))) return;
    setBusy(name);
    try {
      const next = await appApi.forgetSpeaker(name);
      setSpeakers(prev => stableOrder(prev, next));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  const deleteClip = async (name: string, session: number) => {
    if (!window.confirm(t('learned.confirmDeleteClip'))) return;
    setBusy(name);
    try {
      // Deleting a speaker's last sample forgets the speaker entirely: the row simply
      // disappears from the response, which stableOrder handles as a matter of course.
      const next = await appApi.deleteSpeakerClip(name, session);
      setSpeakers(prev => stableOrder(prev, next));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  const reassignClip = async (name: string, session: number, target: string) => {
    setBusy(name);
    try {
      const next = await appApi.reassignSpeakerClip(name, session, target);
      setSpeakers(prev => stableOrder(prev, next));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  // Saved on blur/Enter like the rename box; local drafts so typing does not fire a request per key.
  const [deptDrafts, setDeptDrafts] = useState<Record<string, string>>({});
  const saveDepartment = async (name: string, current: string) => {
    const next = (deptDrafts[name] ?? current).trim();
    if (next === current) return;
    setBusy(name);
    try {
      const res = await appApi.setSpeakerDepartment(name, next);
      setSpeakers(prev => stableOrder(prev, res));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  const setLanguage = async (name: string, language: string) => {
    setBusy(name);
    try {
      const next = await appApi.setSpeakerLanguage(name, language);
      setSpeakers(prev => stableOrder(prev, next));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  const renameSpeaker = async (name: string) => {
    const next = draft.trim();
    setEditing(null);
    if (!next || next === name) return;
    setBusy(name);
    try {
      const res = await appApi.renameSpeaker(name, next);
      setSpeakers(prev => stableOrder(prev, res, { from: name, to: next }));
    } catch (err) {
      if (err instanceof Error && err.message.includes('already exists')) {
        toast.error(t('learned.nameTaken', { name: next }));
      } else {
        fail(err);
      }
      // Reopen the box with what was typed, so a failure does not cost the whole name again.
      setEditing(name);
      setDraft(next);
    } finally {
      setBusy(null);
    }
  };

  // Both sides are editable: the recogniser's mishearing is worth fixing when it was itself a typo,
  // and the replacement is what actually lands in every future transcript.
  const saveCorrection = async (original: string) => {
    setEditingPair(null);
    if (pair.wrong.trim() === original && pair.right.trim() === corrections.find(c => c.wrong === original)?.right) return;
    setBusy(original);
    try {
      setCorrections(await appApi.editCorrection(original, { wrong: pair.wrong.trim(), right: pair.right.trim() }));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  const forgetCorrection = async (wrong: string) => {
    setBusy(wrong);
    try {
      setCorrections(await appApi.forgetCorrection(wrong));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  const clipLabel = (clip: SpeakerClip) =>
    clip.started === null ? t('learned.deletedMeeting') : `${clip.started.slice(0, 10)}・${clip.text ?? ''}`;

  const query = search.trim().toLowerCase();
  const visibleCorrections = query
    ? corrections.filter(c => c.wrong.toLowerCase().includes(query) || c.right.toLowerCase().includes(query))
    : corrections;

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
              <li key={s.name} className={`learned-row${busy === s.name ? ' busy' : ''}`}>
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
                <input
                  className="learned-dept"
                  value={deptDrafts[s.name] ?? s.department}
                  placeholder={t('learned.department')}
                  aria-label={t('learned.department')}
                  title={t('learned.department')}
                  disabled={busy !== null}
                  onChange={e => setDeptDrafts(d => ({ ...d, [s.name]: e.target.value }))}
                  onBlur={() => saveDepartment(s.name, s.department)}
                  onKeyDown={e => e.key === 'Enter' && e.currentTarget.blur()}
                />
                <select
                  className="learned-lang"
                  value={s.language}
                  disabled={busy !== null}
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
                <div className="learned-clips">
                  {s.clip_sessions.map(clip => (
                    <div key={clip.session} className="learned-clip-row">
                      <audio
                        className="learned-clip"
                        controls
                        preload="none"
                        title={clipLabel(clip)}
                        aria-label={clipLabel(clip)}
                        src={`${API_BASE_URL}/speakers/known/${encodeURIComponent(s.name)}/clip?session=${clip.session}`}
                      />
                      <select
                        className="learned-clip-reassign"
                        value=""
                        disabled={busy !== null}
                        aria-label={t('learned.reassignClip')}
                        title={t('learned.reassignClip')}
                        onChange={e => {
                          const value = e.target.value;
                          if (!value) return;
                          if (value === '__new__') {
                            const name = window.prompt(t('learned.newSpeakerName'))?.trim();
                            if (name) reassignClip(s.name, clip.session, name);
                            return;
                          }
                          reassignClip(s.name, clip.session, value);
                        }}
                      >
                        <option value="">{t('learned.reassignClip')}</option>
                        {speakers
                          .filter(o => o.name !== s.name)
                          .map(o => (
                            <option key={o.name} value={o.name}>
                              {o.name}
                            </option>
                          ))}
                        <option value="__new__">{t('learned.reassignNew')}</option>
                      </select>
                      <button
                        className="learned-clip-delete"
                        disabled={busy !== null}
                        title={t('learned.deleteClip')}
                        aria-label={t('learned.deleteClip')}
                        onClick={() => deleteClip(s.name, clip.session)}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  className="learned-forget"
                  disabled={busy !== null}
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
          <>
            <input
              className="learned-search"
              type="search"
              value={search}
              placeholder={t('learned.searchCorrections')}
              aria-label={t('learned.searchCorrections')}
              onChange={e => setSearch(e.target.value)}
            />
            <ul className="learned-list">
              {visibleCorrections.map(c => (
                <li key={c.wrong} className={`learned-row${busy === c.wrong ? ' busy' : ''}`}>
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
                      <button className="learned-save" disabled={busy !== null} title={t('common.save')}>
                        <Check size={16} />
                      </button>
                    </form>
                  ) : (
                    <>
                      <span className="learned-wrong">{c.wrong}</span>
                      <ArrowRight className="learned-arrow" size={14} />
                      <span className="learned-right">{c.right}</span>
                      <span className="etable-count learned-count" title={t('learned.appliedCount', { count: c.count })}>
                        {c.count}
                      </span>
                      <button
                        className="learned-edit"
                        disabled={busy !== null}
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
                    disabled={busy !== null}
                    title={t('learned.forget')}
                    onClick={() => forgetCorrection(c.wrong)}
                  >
                    <Trash2 size={16} />
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
      )}
    </div>
  );
}
