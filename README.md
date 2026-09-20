# recite.net

## Limitations

- Pending actions are not tied to a session, so anyone holding the id can
  confirm one. The ids are random 128-bit values, which is fine for a
  single-user app but not for multi-user.
- Expired pending rows are left in the table rather than swept. Confirming one
  correctly refuses it, but there is no background cleanup.
