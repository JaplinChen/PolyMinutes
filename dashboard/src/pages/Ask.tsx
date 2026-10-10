import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { HelpCircle, Send } from 'lucide-react';
import { Link } from 'react-router-dom';
import { PageHeader } from '../components/PageHeader';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type AskCitation, type AskResult } from '../services/app.api';
import { isSubmitEnter } from '../utils/ime';
import './Ask.css';

// Copied rather than imported from Sessions' TranscriptRow: the same m:ss format, kept local so this
// page does not reach into another page's component for a two-line helper.
const clock = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
};

export function Ask() {
  const { t } = useTranslation();
  useDocumentTitle(t('ask.title'));
  const toast = useToast();

  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AskResult | null>(null);

  // Citations arrive flat but read better grouped: one session, then the utterances it contributed.
  const grouped = useMemo(() => {
    if (!result) return [] as { session: number; citations: AskCitation[] }[];
    const order: number[] = [];
    const by = new Map<number, AskCitation[]>();
    for (const c of result.citations) {
      if (!by.has(c.session_id)) {
        by.set(c.session_id, []);
        order.push(c.session_id);
      }
      by.get(c.session_id)!.push(c);
    }
    return order.map(session => ({ session, citations: by.get(session)! }));
  }, [result]);

  const ask = async () => {
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);
    try {
      setResult(await appApi.ask(q));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ask-page">
      <PageHeader title={t('ask.title')} subtitle={t('ask.subtitle')} />

      <section className="ask-panel">
        <textarea
          className="ask-input"
          value={question}
          disabled={busy}
          aria-label={t('ask.title')}
          placeholder={t('ask.placeholder')}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => {
            if (isSubmitEnter(e) && !e.shiftKey) {
              e.preventDefault();
              ask();
            }
          }}
        />
        <button className="btn-primary ask-submit" onClick={ask} disabled={busy || !question.trim()}>
          <Send size={16} />
          {busy ? t('ask.submitting') : t('ask.submit')}
        </button>
      </section>

      {!result && !busy && (
        <div className="ask-empty">
          <HelpCircle size={32} strokeWidth={1} />
          <p className="ask-hint">{t('ask.hint')}</p>
          <p className="ask-hint-example">{t('ask.hintExample')}</p>
        </div>
      )}

      {result && (
        <section className="ask-panel ask-result">
          {!result.verified && <p className="ask-notice ask-unverified" role="alert">{t('ask.unverified')}</p>}

          <h3 className="ask-heading">{t('ask.answerHeading')}</h3>
          {result.answer.trim() ? (
            <p className="ask-answer">{result.answer}</p>
          ) : (
            <p className="ask-answer ask-muted">{t('ask.noAnswer')}</p>
          )}

          {result.truncated.length > 0 && (
            <p className="ask-notice ask-truncated">{t('ask.truncated', { count: result.truncated.length })}</p>
          )}

          {grouped.length > 0 && (
            <>
              <h3 className="ask-heading">{t('ask.citationsHeading')}</h3>
              <div className="ask-citations">
                {grouped.map(({ session, citations }) => (
                  <div key={session} className="ask-citation-group">
                    {citations.map(c => (
                      <Link
                        key={c.line_id}
                        className="ask-citation"
                        to={`/sessions?session=${c.session_id}&line=${c.line_id}`}
                      >
                        <span className="ask-citation-meta">
                          <span className="ask-citation-speaker">{c.speaker}</span>
                          <span className="ask-citation-time">{clock(c.start)}</span>
                        </span>
                        <span className="ask-citation-text">{c.text}</span>
                      </Link>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}

          {result.dropped_citations > 0 && (
            <p className="ask-footnote">{t('ask.dropped', { count: result.dropped_citations })}</p>
          )}
        </section>
      )}
    </div>
  );
}

export default Ask;
