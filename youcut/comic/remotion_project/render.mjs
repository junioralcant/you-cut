#!/usr/bin/env node
// Entrypoint Node CLI do engine `youcut comic --engine remotion`.
//
// Uso: node render.mjs --props <props.json> --out <output.mp4> [--composition ComicVideo]
//
// Lê os InputProps de um arquivo JSON e renderiza via @remotion/renderer
// emitindo progresso como JSON-lines em stdout (`{"progress": 0.42}`),
// para que o orquestrador Python `RemotionRenderer` possa repassar ao CLI.

import { bundle } from "@remotion/bundler";
import { renderMedia, selectComposition } from "@remotion/renderer";
import { copyFileSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function parseArgs(argv) {
  const args = { composition: "ComicVideo", concurrency: null };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    const next = argv[i + 1];
    switch (arg) {
      case "--props":
        args.props = next;
        i++;
        break;
      case "--out":
        args.out = next;
        i++;
        break;
      case "--composition":
        args.composition = next;
        i++;
        break;
      case "--concurrency":
        args.concurrency = Number(next);
        i++;
        break;
      case "--help":
      case "-h":
        args.help = true;
        break;
      default:
        // ignore unknown flags
        break;
    }
  }
  return args;
}

function emit(record) {
  process.stdout.write(JSON.stringify(record) + "\n");
}

function fail(reason) {
  emit({ error: reason });
  process.exit(1);
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(
      "Usage: node render.mjs --props <props.json> --out <output.mp4> [--composition ComicVideo] [--concurrency N]",
    );
    return;
  }
  if (!args.props || !args.out) {
    fail("--props and --out are required");
    return;
  }

  const propsPath = resolve(args.props);
  const outPath = resolve(args.out);

  let inputProps;
  try {
    inputProps = JSON.parse(readFileSync(propsPath, "utf8"));
  } catch (err) {
    fail(`failed to read props: ${err.message}`);
    return;
  }

  // Cria um publicDir isolado com cópias dos assets referenciados pelos
  // props (audio + futuras anchors / mouth sheets). Usa cópia em vez de
  // symlink porque o bundler do Remotion não segue symlinks de maneira
  // consistente entre plataformas.
  const publicDir = mkdtempSync(join(tmpdir(), "youcut-remotion-public-"));
  const propsDir = dirname(propsPath);
  const linkAsset = (rawPath) => {
    if (!rawPath) return rawPath;
    const target = isAbsolute(rawPath)
      ? resolve(rawPath)
      : resolve(propsDir, rawPath);
    const linkName = basename(target);
    const linkPath = join(publicDir, linkName);
    copyFileSync(target, linkPath);
    return linkName;
  };

  if (inputProps.audio_path) {
    inputProps.audio_path = linkAsset(inputProps.audio_path);
  }
  if (inputProps.characters && typeof inputProps.characters === "object") {
    for (const charId of Object.keys(inputProps.characters)) {
      const c = inputProps.characters[charId] ?? {};
      if (c.anchor_path) c.anchor_path = linkAsset(c.anchor_path);
      if (c.mouth_sheet_path) c.mouth_sheet_path = linkAsset(c.mouth_sheet_path);
    }
  }

  emit({ stage: "bundle", path: propsPath });
  const entry = resolve(__dirname, "src/index.ts");
  const bundleLocation = await bundle({
    entryPoint: entry,
    publicDir,
    webpackOverride: (config) => config,
  });

  emit({ stage: "select_composition", id: args.composition });
  const composition = await selectComposition({
    serveUrl: bundleLocation,
    id: args.composition,
    inputProps,
  });

  emit({
    stage: "render",
    fps: composition.fps,
    duration_in_frames: composition.durationInFrames,
    width: composition.width,
    height: composition.height,
  });

  let lastReportedProgress = -1;
  await renderMedia({
    composition,
    serveUrl: bundleLocation,
    codec: "h264",
    pixelFormat: "yuv420p",
    outputLocation: outPath,
    inputProps,
    concurrency: args.concurrency ?? null,
    onProgress: ({ progress }) => {
      const rounded = Math.round(progress * 100) / 100;
      if (rounded !== lastReportedProgress) {
        lastReportedProgress = rounded;
        emit({ progress: rounded });
      }
    },
  });

  emit({ stage: "done", output: outPath });
}

main().catch((err) => {
  fail(err && err.stack ? err.stack : String(err));
});
