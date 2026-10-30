import { ref } from 'vue';

interface Theme {
  id: string;
  name: string;
  scheme: 'dark' | 'light';
  colors: Record<string, string>;
}

export const themes = Object.values(
  import.meta.glob<Theme>('./themes/*.json', {
    eager: true,
    import: 'default',
  }),
);
export const themeId = ref('paper');
let selected = false;

export function applyTheme(id: string, remember = false): void {
  const theme =
    themes.find((item) => item.id === id) ??
    themes.find((item) => item.id === 'paper') ??
    themes[0];
  for (const [key, value] of Object.entries(theme.colors)) {
    document.documentElement.style.setProperty(`--${key}`, value);
  }
  document.documentElement.style.colorScheme = theme.scheme;
  document.documentElement.dataset.theme = theme.id;
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme.colors.bg);
  themeId.value = theme.id;
  if (remember) {
    selected = true;
    localStorage.setItem('ninfer.theme', theme.id);
  }
}

export function initializeTheme(defaultTheme = 'paper'): void {
  if (selected) return;
  const stored = localStorage.getItem('ninfer.theme');
  if (stored && !themes.some((theme) => theme.id === stored)) {
    localStorage.removeItem('ninfer.theme');
  }
  applyTheme(localStorage.getItem('ninfer.theme') ?? defaultTheme);
}
