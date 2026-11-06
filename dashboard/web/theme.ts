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
export const themeId = ref(themes[0]?.id ?? '');
let selected = false;

export function applyTheme(id: string, remember = false): void {
  const theme = themes.find((item) => item.id === id) ?? themes[0];
  if (!theme) return;
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

export function initializeTheme(defaultTheme?: string): void {
  if (selected) return;
  const stored = localStorage.getItem('ninfer.theme');
  if (stored && !themes.some((theme) => theme.id === stored)) {
    localStorage.removeItem('ninfer.theme');
  }
  const target = localStorage.getItem('ninfer.theme') ?? defaultTheme ?? themes[0]?.id;
  if (target) applyTheme(target);
}
