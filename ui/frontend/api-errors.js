// Turn an API error body into readable text.
//
// FastAPI answers a rejected request with a structured detail: a string for handled
// errors, or a list of {loc, msg} objects for validation errors. Stringifying that list
// produced "[object Object]", which told an operator nothing, so every entry is rendered
// as its field path and message. Only those two fields, plus a short list of known
// message keys, are ever shown: a raw exception body must not reach the banner.
const MAX_FIELD_ERRORS = 8;
const MAX_LENGTH = 600;

function primitive(value) {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return null;
}

function firstPrimitive(...values) {
  for (const value of values) {
    const text = primitive(value);
    if (text !== null && text.trim()) return text.trim();
  }
  return null;
}

function fieldPath(loc) {
  if (!Array.isArray(loc)) return '';
  const parts = [];
  for (const part of loc) {
    if (Number.isInteger(part) && parts.length) {
      parts[parts.length - 1] += `[${part}]`;
      continue;
    }
    const name = primitive(part);
    if (name !== null && name) parts.push(name);
  }
  // FastAPI prefixes body locations with "body"; the request itself is the body.
  if (parts[0] === 'body') parts.shift();
  return parts.join('.');
}

function entryText(item) {
  if (typeof item === 'string') return item.trim() || null;
  if (!item || typeof item !== 'object' || Array.isArray(item)) return null;
  const path = fieldPath(item.loc);
  const message = firstPrimitive(item.msg, item.message, item.error);
  if (path && message) return `${path}: ${message}`;
  if (message) return message;
  return path ? `${path}: invalid value` : null;
}

function knownDetail(payload) {
  const detail = payload.detail;
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null;
  return firstPrimitive(detail.message, detail.msg, detail.error, detail.reason, detail.detail);
}

function clip(text) {
  return text.length > MAX_LENGTH ? `${text.slice(0, MAX_LENGTH - 1)}\u2026` : text;
}

export function apiErrorMessage(payload, fallback) {
  const detail = Array.isArray(payload) ? payload : (payload && typeof payload === 'object' ? payload.detail : null);
  if (Array.isArray(detail)) {
    const parts = [];
    for (const item of detail) {
      const text = entryText(item);
      if (text && !parts.includes(text)) parts.push(text);
    }
    if (parts.length) {
      const shown = parts.slice(0, MAX_FIELD_ERRORS);
      const more = parts.length - shown.length;
      const suffix = more ? `; and ${more} more ${more === 1 ? 'error' : 'errors'}` : '';
      return clip(`Invalid request: ${shown.join('; ')}${suffix}`);
    }
  }
  const direct = (typeof payload === 'string' ? payload.trim() : null) ?? primitive(payload)
    ?? (payload && typeof payload === 'object' ? knownDetail(payload) : null)
    ?? (payload && typeof payload === 'object' && typeof payload.detail === 'string' ? payload.detail.trim() : null);
  if (direct) return clip(direct);
  const text = typeof fallback === 'string' ? fallback.trim() : '';
  return clip(text || 'Request failed');
}
