import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type { PendingRule } from '../../services/app.api';
import './PendingRuleDialog.css';

interface Props {
  rule: PendingRule;
  onLearn: () => void;
  onSkip: () => void;
}

// One rule at a time: each is its own judgement about whether a literal swap is safe everywhere.
export function PendingRuleDialog({ rule, onLearn, onSkip }: Props) {
  const { t } = useTranslation();
  const skipRef = useRef<HTMLButtonElement>(null);

  // Focus lands on skip: the dialog opens right after Enter saved the line, and a second Enter out
  // of habit must not teach a rule that rewrites every other line.
  useEffect(() => {
    skipRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onSkip(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [rule, onSkip]);

  return (
    <div className="prule-backdrop">
      <div className="prule-dialog" role="dialog" aria-modal="true" aria-labelledby="prule-title">
        <h3 id="prule-title" className="prule-title">{t('sessions.pendingRuleTitle')}</h3>
        <p className="prule-body">
          {t('sessions.pendingRuleBody', { wrong: rule.wrong, right: rule.right, count: rule.count })}
        </p>
        <ul className="prule-examples">
          {rule.examples.map((ex, i) => <li key={i}>{ex}</li>)}
        </ul>
        <div className="prule-actions">
          <button type="button" className="prule-skip" ref={skipRef} onClick={onSkip}>
            {t('sessions.pendingRuleSkip')}
          </button>
          <button type="button" className="prule-learn" onClick={onLearn}>
            {t('sessions.pendingRuleLearn')}
          </button>
        </div>
      </div>
    </div>
  );
}
