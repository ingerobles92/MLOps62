#!/usr/bin/env bash
set -euo pipefail

confirm() {
  local prompt="${1:-Are you sure?} [y/N] "
  read -r -p "$prompt" ans || true
  case "${ans:-}" in
    y|Y|yes|YES) return 0 ;;
    *)           return 1 ;;
  esac
}

if ! command -v dvc >/dev/null 2>&1; then
  echo "❌ dvc no está instalado en este entorno." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "❌ git no está instalado en este entorno." >&2
  exit 1
fi
if [ ! -d .git ] || [ ! -d .dvc ]; then
  echo "❌ Debes ejecutar este script en la raíz del repo (donde existen .git y .dvc)." >&2
  exit 1
fi

echo "📦 Repo: $(pwd)"
echo "🗂  Branch actual: $(git rev-parse --abbrev-ref HEAD)"

echo
echo "➡️  Ingresa una o varias rutas de CSV (separadas por espacio)."
echo "   Ej: data/processed/work_absenteeism_clean_v1.0.csv data/interim/foo.csv"
read -r -p "Rutas: " -a CSV_PATHS

if [ "${#CSV_PATHS[@]}" -eq 0 ]; then
  echo "❌ No se proporcionaron rutas. Saliendo."
  exit 1
fi

MISSING=0
for p in "${CSV_PATHS[@]}"; do
  if [ ! -f "$p" ]; then
    echo "❌ No existe: $p"
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "❌ Corrige las rutas inexistentes y vuelve a intentar."
  exit 1
fi

echo
echo "➡️  Ejecutar DVC ADD sobre las rutas:"
for p in "${CSV_PATHS[@]}"; do
  echo "   - $p"
done

if confirm "¿Deseas ejecutar dvc add ahora?"; then
  for p in "${CSV_PATHS[@]}"; do
    echo "🔧 dvc add $p"
    dvc add "$p"
  done
  echo "✅ dvc add completado."
else
  echo "⏭️  Omitido dvc add por elección del usuario."
fi

POINTERS=()
for p in "${CSV_PATHS[@]}"; do
  if [ -f "${p}.dvc" ]; then
    POINTERS+=("${p}.dvc")
  fi
done

echo
if [ "${#POINTERS[@]}" -eq 0 ]; then
  echo "ℹ️  No hay punteros .dvc que agregar a Git (quizá omitiste dvc add)."
else
  echo "🧾 Punteros .dvc detectados para Git:"
  for d in "${POINTERS[@]}"; do
    echo "   - $d"
  done
fi

if confirm "¿Deseas subir los blobs a S3 con dvc push ahora?"; then
  echo "☁️  dvc push (esto puede tardar si los archivos son grandes)…"
  dvc push -r s3remote
  echo "✅ dvc push completado."
else
  echo "⏭️  Omitido dvc push por elección del usuario."
fi

if [ "${#POINTERS[@]}" -gt 0 ] && confirm "¿Deseas subir los punteros .dvc a GitHub (git add/commit/push)?"; then
  git add "${POINTERS[@]}"

  echo
  echo "📋 Cambios preparados:"
  git status --short || true

  read -r -p "Mensaje de commit [data: update datasets]: " COMMIT_MSG
  COMMIT_MSG="${COMMIT_MSG:-data: update datasets}"

  if git diff --cached --quiet; then
    echo "ℹ️  No hay cambios nuevos para commitear."
  else
    git commit -m "$COMMIT_MSG"
    echo "✅ commit realizado."
  fi

  CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  read -r -p "Remote para push [origin]: " REMOTE
  REMOTE="${REMOTE:-origin}"

  echo "⬆️  git push ${REMOTE} ${CUR_BRANCH}"
  git push "$REMOTE" "$CUR_BRANCH"
  echo "✅ git push completado."
else
  echo "⏭️  Omitido git add/commit/push por elección del usuario."
fi

echo
echo "🎉 Listo. Resumen:"
echo "   - dvc add: ejecutado según tu elección."
echo "   - dvc push: ejecutado según tu elección."
echo "   - git push: ejecutado según tu elección."
