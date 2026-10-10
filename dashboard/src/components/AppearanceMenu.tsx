import { useRef, type CSSProperties } from 'react';
import { useTranslation } from 'react-i18next';
import { Sun, Moon, Monitor, Palette } from 'lucide-react';
import { useTheme, type Theme } from '../hooks/useTheme';
import { useDismissableMenu } from '../hooks/useDismissableMenu';

const themeIcons = { light: Sun, dark: Moon, system: Monitor, anthropic: Palette, 'anthropic-dark': Palette };
const modes: Theme[] = ['light', 'dark', 'system', 'anthropic', 'anthropic-dark'];

export function AppearanceMenu() {
  const { t } = useTranslation();
  const { theme, setTheme, palette, setPalette, paletteOptions } = useTheme();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const { open, setOpen, ref } = useDismissableMenu<HTMLDivElement>(() => triggerRef.current?.focus());

  const ThemeIcon = themeIcons[theme];
  const themeLabel = t(`theme.${theme}`);
  const activePalette = paletteOptions.find(option => option.value === palette) ?? paletteOptions[0];

  return (
    <div className="appearance-menu" ref={ref}>
      <button
        ref={triggerRef}
        className="theme-toggle-btn icon-only"
        onClick={() => setOpen(value => !value)}
        title={t('theme.label', { value: themeLabel })}
        aria-label={t('theme.appearance')}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span
          className="appearance-button-cue"
          style={{ '--swatch-color': activePalette.color } as CSSProperties}
          aria-hidden="true"
        >
          <ThemeIcon size={14} />
        </span>
      </button>
      {open && (
        <div className="appearance-menu-list" role="menu" aria-label={t('theme.appearance')}>
          <div className="appearance-menu-header">
            <div>
              <strong>{t('theme.appearance')}</strong>
              <span>{activePalette.label}</span>
            </div>
            <span
              className="appearance-current-swatch"
              style={{ '--swatch-color': activePalette.color } as CSSProperties}
              aria-hidden="true"
            />
          </div>
          <div className="appearance-section">
            <span className="appearance-section-label">{t('theme.mode')}</span>
            <div className="appearance-mode-grid">
              {modes.map(mode => {
                const ModeIcon = themeIcons[mode];
                return (
                  <button
                    key={mode}
                    className={`appearance-mode ${theme === mode ? 'active' : ''}`}
                    onClick={() => setTheme(mode)}
                    type="button"
                    role="menuitemradio"
                    aria-checked={theme === mode}
                  >
                    <ModeIcon size={16} />
                    <span>{t(`theme.${mode}`)}</span>
                  </button>
                );
              })}
            </div>
          </div>
          <div className="appearance-section">
            <span className="appearance-section-label">{t('theme.palette')}</span>
            <div className="palette-grid">
              {paletteOptions.map(option => (
                <button
                  key={option.value}
                  className={`palette-swatch ${palette === option.value ? 'active' : ''}`}
                  onClick={() => setPalette(option.value)}
                  type="button"
                  title={option.label}
                  role="menuitemradio"
                  aria-checked={palette === option.value}
                  style={{ '--swatch-color': option.color } as CSSProperties}
                >
                  <span />
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
