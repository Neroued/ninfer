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
export const themeId = ref('ember');
let selected = false;

export function applyTheme(id: string, remember = false): void {
  const theme = themes.find((item) => item.id === id) ?? themes[0];
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

export function initializeTheme(defaultTheme = 'ember'): void {
  if (!selected) applyTheme(localStorage.getItem('ninfer.theme') ?? defaultTheme);
}
