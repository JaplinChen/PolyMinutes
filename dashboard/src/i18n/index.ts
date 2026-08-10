import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import en from './locales/en.json';
import zhHK from './locales/zh-HK.json';
import vi from './locales/vi.json';

export const supportedLanguages = ['en', 'zh-HK', 'vi'] as const;
export type SupportedLanguage = (typeof supportedLanguages)[number];

// Kept (empty) so Layout's direction logic stays intact if an RTL locale is ever added back.
export const rtlLanguages: SupportedLanguage[] = [];

export const languageOptions: Array<{ value: SupportedLanguage; label: string; compactLabel: string }> = [
  { value: 'zh-HK', label: '繁體中文', compactLabel: '繁中' },
  { value: 'vi', label: 'Tiếng Việt', compactLabel: 'VI' },
  { value: 'en', label: 'English', compactLabel: 'EN' },
];

export function resolveSupportedLanguage(lang?: string): SupportedLanguage {
  const value = lang || 'zh-HK';
  const exact = supportedLanguages.find(supported => supported.toLowerCase() === value.toLowerCase());
  if (exact) return exact;

  const base = value.toLowerCase().split('-')[0];
  // Any Chinese subtag maps to 繁體 — this app has no 简体 locale.
  if (base === 'zh') return 'zh-HK';

  return supportedLanguages.find(supported => supported === base) ?? 'en';
}

void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      'zh-HK': { translation: zhHK },
      vi: { translation: vi },
    },
    fallbackLng: 'en',
    // A key missing from zh-HK or vi falls back to English and looks merely odd; a key missing
    // from en renders as the literal `sessions.refining` on screen. Both have shipped here before.
    // Dev-only so a production build never pays for the check.
    saveMissing: import.meta.env.DEV,
    missingKeyHandler: import.meta.env.DEV
      ? (lngs, ns, key) => console.warn(`[i18n] missing key "${key}" for ${lngs.join(', ')} (${ns})`)
      : undefined,
    supportedLngs: supportedLanguages as unknown as string[],
    nonExplicitSupportedLngs: false,
    interpolation: { escapeValue: false },
    detection: {
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: 'polyminutes_language',
      caches: ['localStorage'],
      convertDetectedLanguage: (lang: string) => resolveSupportedLanguage(lang),
    },
    react: { useSuspense: false },
  });

function applyDirection(lang: string) {
  const resolved = resolveSupportedLanguage(lang);
  const dir = rtlLanguages.includes(resolved) ? 'rtl' : 'ltr';
  if (typeof document !== 'undefined') {
    document.documentElement.lang = resolved;
    document.documentElement.dir = dir;
  }
}

applyDirection(i18n.language);
i18n.on('languageChanged', applyDirection);

export default i18n;
