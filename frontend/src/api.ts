declare global {interface Window {STUDIO_TOKEN: string}}

export async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch('/api' + path, {method, headers: {'Content-Type': 'application/json', 'X-Studio-Token': window.STUDIO_TOKEN}, body: body === undefined ? undefined : JSON.stringify(body)});
  if (!response.ok) {const value = await response.json().catch(() => ({})); throw new Error(typeof value.detail === 'string' ? value.detail : 'The request could not be completed.');}
  return response.json();
}
export const busy = (job?: {status:string}) => !!job && ['queued','running','pausing'].includes(job.status);
export const lang = (code: string) => {try {return new Intl.DisplayNames(['en'], {type:'language'}).of(code) || code;} catch {return code;}};
