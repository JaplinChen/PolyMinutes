import { useState, useEffect, useRef } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { BookMarked, Sprout, FileText, Mic, Settings, Menu, X, ChevronLeft, ChevronRight, MessageCircleQuestion } from 'lucide-react';
import { resolveSupportedLanguage, rtlLanguages } from '../i18n';
import { healthApi } from '../services/api';
import { LanguageMenu } from './LanguageMenu';
import { AppearanceMenu } from './AppearanceMenu';
import './Layout.css';

const navItems = [
  { to: '/capture', icon: Mic, key: 'capture' as const },
  { to: '/sessions', icon: FileText, key: 'sessions' as const },
  { to: '/ask', icon: MessageCircleQuestion, key: 'ask' as const },
  { to: '/glossary', icon: BookMarked, key: 'glossary' as const },
  { to: '/learned', icon: Sprout, key: 'learned' as const },
  { to: '/settings', icon: Settings, key: 'settings' as const },
];

export function Layout() {
  const { t, i18n } = useTranslation();
  const { pathname } = useLocation();
  const mainRef = useRef<HTMLElement>(null);

  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  // Show the build-time version immediately, then replace it with the live running version from the
  // backend so a stale-built bundle can't display the wrong number. Falls back silently on error.
  const [version, setVersion] = useState(__APP_VERSION__);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) setIsMobileOpen(false);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    let active = true;
    healthApi
      .check()
      .then(info => {
        if (active && info?.version) setVersion(info.version);
      })
      .catch(() => {
        /* keep the build-time fallback */
      });
    return () => {
      active = false;
    };
  }, []);

  const handleNavClick = () => {
    if (isMobile) setIsMobileOpen(false);
  };

  useEffect(() => {
    document.body.style.overflow = isMobileOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobileOpen]);

  useEffect(() => {
    const saved = parseInt(localStorage.getItem('sidebarWidth') || '', 10);
    if (saved >= 180 && saved <= 480) document.documentElement.style.setProperty('--sidebar-w', `${saved}px`);
  }, []);

  // <main> is the scroll container and it outlives the route, so its offset would otherwise carry
  // over: leave one page halfway down, arrive at the next one already past its heading.
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0 });
  }, [pathname]);

  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    document.body.classList.add('sidebar-resizing');
    const onMove = (ev: MouseEvent) => {
      const w = Math.min(480, Math.max(180, isRtl ? window.innerWidth - ev.clientX : ev.clientX));
      document.documentElement.style.setProperty('--sidebar-w', `${w}px`);
    };
    const onUp = () => {
      document.body.classList.remove('sidebar-resizing');
      localStorage.setItem(
        'sidebarWidth',
        String(parseInt(document.documentElement.style.getPropertyValue('--sidebar-w'), 10) || 260)
      );
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  const toggleCollapse = () => setIsCollapsed(!isCollapsed);
  const toggleMobile = () => setIsMobileOpen(!isMobileOpen);

  const currentLang = resolveSupportedLanguage(i18n.resolvedLanguage || i18n.language);
  const isRtl = rtlLanguages.includes(currentLang);

  return (
    <div className="layout">
      {isMobile && (
        <header className="mobile-header">
          <button className="mobile-menu-btn" onClick={toggleMobile} aria-label={t('common.expand')}>
            {isMobileOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          <div className="mobile-brand">
            <img src="/favicon.svg" alt="PolyMinutes" className="sidebar-logo" />
            <span className="brand-name">{t('common.appName')}</span>
          </div>
          <div style={{ width: 40 }} />
        </header>
      )}

      {isMobile && isMobileOpen && <div className="sidebar-overlay" onClick={() => setIsMobileOpen(false)} />}

      <aside
        className={`sidebar ${isCollapsed ? 'collapsed' : ''} ${isMobile ? 'mobile' : ''} ${isMobileOpen ? 'open' : ''}`}
      >
        <div className="sidebar-header">
          <img src="/favicon.svg" alt="PolyMinutes" className="sidebar-logo" />
          {!isCollapsed && (
            <div className="sidebar-brand">
              <span className="brand-name">{t('common.appName')}</span>
              <span className="brand-version">v{version}</span>
            </div>
          )}
        </div>

        {!isMobile && !isCollapsed && <div className="sidebar-resizer" onMouseDown={startResize} />}

        {!isMobile && (
          <button
            className="collapse-toggle"
            onClick={toggleCollapse}
            title={isCollapsed ? t('common.expand') : t('common.collapse')}
            aria-label={isCollapsed ? t('common.expand') : t('common.collapse')}
          >
            {isCollapsed ? (
              isRtl ? (
                <ChevronLeft size={16} />
              ) : (
                <ChevronRight size={16} />
              )
            ) : isRtl ? (
              <ChevronRight size={16} />
            ) : (
              <ChevronLeft size={16} />
            )}
          </button>
        )}

        <nav className="sidebar-nav">
          {navItems.map(({ to, icon: Icon, key }) => {
            const label = t(`nav.${key}`);
            return (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                end={to === '/'}
                onClick={handleNavClick}
                title={isCollapsed ? label : undefined}
              >
                <Icon size={20} />
                {!isCollapsed && <span>{label}</span>}
              </NavLink>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <LanguageMenu />
          <AppearanceMenu />
        </div>
      </aside>

      <main ref={mainRef} className={`main-content ${isCollapsed ? 'expanded' : ''} ${isMobile ? 'mobile' : ''}`}>
        <Outlet />
      </main>
    </div>
  );
}
