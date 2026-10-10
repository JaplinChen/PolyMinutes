import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Languages } from 'lucide-react';
import { languageOptions, resolveSupportedLanguage, type SupportedLanguage } from '../i18n';
import { useDismissableMenu } from '../hooks/useDismissableMenu';

export function LanguageMenu() {
  const { t, i18n } = useTranslation();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const { open, setOpen, ref } = useDismissableMenu<HTMLDivElement>(() => triggerRef.current?.focus());

  const currentLang = resolveSupportedLanguage(i18n.resolvedLanguage || i18n.language);
  const currentLangOption = languageOptions.find(option => option.value === currentLang);
  const languageLabel = currentLangOption?.label ?? 'English';

  const changeLanguage = (language: SupportedLanguage) => {
    setOpen(false);
    triggerRef.current?.focus();
    void i18n.changeLanguage(language);
  };

  return (
    <div className="language-menu" ref={ref}>
      <button
        ref={triggerRef}
        className="theme-toggle-btn icon-only"
        onClick={() => setOpen(value => !value)}
        title={`${t('common.language')}: ${languageLabel}`}
        aria-label={t('common.language')}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <Languages size={18} />
      </button>
      {open && (
        <div className="language-menu-list" role="menu" aria-label={t('common.language')}>
          {languageOptions.map(option => (
            <button
              key={option.value}
              className={`language-menu-item ${option.value === currentLang ? 'active' : ''}`}
              onClick={() => changeLanguage(option.value)}
              role="menuitemradio"
              aria-checked={option.value === currentLang}
            >
              <span>{option.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
