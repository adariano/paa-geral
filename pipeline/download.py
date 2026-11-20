"""Download genérico de arquivos de corpus por URL HTTP(S).

O script não escolhe nem valida a licença de uma base específica. O grupo deve
registrar a fonte, licença, data de acesso e URL antes de iniciar o download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from tqdm import tqdm


def safe_filename(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if not name:
        raise ValueError(f"Não foi possível inferir o nome do arquivo a partir da URL: {url}")
    return Path(name).name


def download_file(url: str, destination: Path, timeout: int = 60) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "PAA-Historia-Brasileira/1.0"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    downloaded = 0

    try:
        with urlopen(request, timeout=timeout) as response:
            total_header = response.headers.get("Content-Length")
            total_bytes = int(total_header) if total_header and total_header.isdigit() else None
            with temporary.open("wb") as output:
                with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=destination.name) as bar:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        downloaded += len(block)
                        bar.update(len(block))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "url": url,
        "filename": destination.name,
        "bytes": downloaded,
        "sha256": digest.hexdigest(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa arquivos de uma fonte HTTP(S) para data/raw.")
    parser.add_argument("--url", action="append", required=True, help="URL direta; pode ser repetida")
    parser.add_argument("--filename", action="append", help="Nome destino correspondente a cada --url")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--source-name", default="Não informado")
    parser.add_argument("--license", dest="license_name", default="VERIFICAR LICENÇA")
    args = parser.parse_args()

    if args.filename and len(args.filename) != len(args.url):
        parser.error("Informe exatamente um --filename para cada --url, ou não informe nenhum.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "download_manifest.json"
    entries = []
    for index, url in enumerate(args.url):
        if urlparse(url).scheme not in {"http", "https"}:
            parser.error(f"URL precisa usar HTTP ou HTTPS: {url}")
        filename = args.filename[index] if args.filename else safe_filename(url)
        if Path(filename).name != filename:
            parser.error("O filename deve ser um nome simples, sem diretórios.")
        destination = args.output_dir / filename
        print(f"[INFO] Baixando {url}")
        entries.append(download_file(url, destination, timeout=args.timeout))

    manifest = {
        "source_name": args.source_name,
        "license": args.license_name,
        "downloads": entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total_bytes = sum(int(item["bytes"]) for item in entries)
    print(f"[OK] {len(entries)} arquivo(s), {total_bytes:,} bytes baixados.")
    print(f"[INFO] Manifesto salvo em: {manifest_path}")
    print("[ATENÇÃO] Tamanho baixado não é sinônimo de 10 GiB de texto puro; confira com report_corpus.py.")


if __name__ == "__main__":
    main()
