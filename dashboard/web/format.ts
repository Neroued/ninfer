export function number(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value)
    ? '—'
    : value.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
}

export function tokens(value: number | null | undefined): string {
  if (value == null) return '—';
  return value >= 1e6
    ? `${number(value / 1e6, 2)}m`
    : value >= 1e3
      ? `${number(value / 1e3, 1)}k`
      : number(value, 0);
}

export function ms(value: number | null | undefined): string {
  if (value == null) return '—';
  return value >= 1000 ? `${number(value / 1000, 2)} s` : `${number(value, value < 10 ? 2 : 1)} ms`;
}

export function bytes(value: number | null | undefined, digits = 1): string {
  if (value == null) return '—';
  if (value >= 1024 ** 3) return `${number(value / 1024 ** 3, digits)} GiB`;
  if (value >= 1024 ** 2) return `${number(value / 1024 ** 2, digits)} MiB`;
  return `${number(value / 1024, digits)} KiB`;
}

export function age(seconds: number | null | undefined): string {
  if (seconds == null) return '—';
  if (seconds < 60) return `${number(Math.max(0, seconds), 1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  if (seconds < 86400)
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}

export function sampleAge(milliseconds: number | undefined): string {
  return age(milliseconds === undefined ? null : milliseconds / 1000);
}

export function clock(at: number, full = false): string {
  return new Date(at).toLocaleString(
    'en-GB',
    full
      ? { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }
      : { hour: '2-digit', minute: '2-digit', second: '2-digit' },
  );
}

export function dateInput(at: number): string {
  const date = new Date(at);
  return new Date(at - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
}

export function label(value: string): string {
  return value.replaceAll('_', ' ');
}
