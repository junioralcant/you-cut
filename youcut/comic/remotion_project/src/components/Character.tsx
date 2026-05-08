// Character — exibe a célula correta da mouth sheet baseado em `lipSync`,
// com idle blink (filter brightness pulse) e breathe (oscilação Y).
//
// O Character é tipicamente posicionado no centro do canvas pelo Scene
// (override possível via prop `position`).

import { useCurrentFrame, useVideoConfig, staticFile } from "remotion";
import type { CharacterAssets, MouthEvent, MouthShape } from "../types";

const DEFAULT_BLINK_PERIOD_SEC = 4.5;
const BLINK_DURATION_SEC = 0.12;
const BLINK_BRIGHTNESS = 0.7;
const BREATHE_AMPLITUDE_PX = 2;
const BREATHE_PERIOD_SEC = 3.0;

const FALLBACK_CELLS: Record<MouthShape, [number, number, number, number]> = {
  closed: [0, 0, 512, 512],
  open_mid: [512, 0, 1024, 512],
  open_wide: [0, 512, 512, 1024],
  open_round: [512, 512, 1024, 1024],
};

interface CharacterProps {
  characterId: string;
  assets: CharacterAssets;
  lipSync: MouthEvent[];
  blinkPeriodSec?: number;
  position?: { left: number; top: number };
  cellSize?: number;
}

const findShapeAt = (
  lipSync: MouthEvent[],
  characterId: string,
  t: number,
): MouthShape => {
  for (const ev of lipSync) {
    if (ev.character_id !== characterId) continue;
    if (t >= ev.start_sec && t < ev.end_sec) return ev.shape;
  }
  return "closed";
};

export const Character: React.FC<CharacterProps> = ({
  characterId,
  assets,
  lipSync,
  blinkPeriodSec = DEFAULT_BLINK_PERIOD_SEC,
  position,
  cellSize,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const t = frame / fps;

  const shape = findShapeAt(lipSync, characterId, t);
  const cells = assets.cells ?? FALLBACK_CELLS;
  const [x1, y1, x2, y2] = cells[shape] ?? FALLBACK_CELLS[shape];

  const sheetW = Math.max(...Object.values(cells).map((c) => c[2]));
  const sheetH = Math.max(...Object.values(cells).map((c) => c[3]));
  const cellW = x2 - x1;
  const cellH = y2 - y1;

  const renderSize = cellSize ?? cellW;
  const scale = renderSize / cellW;

  const breatheY =
    Math.sin((t / BREATHE_PERIOD_SEC) * 2 * Math.PI) * BREATHE_AMPLITUDE_PX;

  const blinkPhase = blinkPeriodSec > 0 ? t % blinkPeriodSec : Number.POSITIVE_INFINITY;
  const isBlinking = blinkPhase < BLINK_DURATION_SEC;
  const brightness = isBlinking ? BLINK_BRIGHTNESS : 1;

  const left = position?.left ?? Math.round(width / 2 - renderSize / 2);
  const top = position?.top ?? Math.round(height / 2 - renderSize / 2);

  if (!assets.mouth_sheet_path) {
    return null;
  }

  return (
    <div
      data-character-id={characterId}
      data-mouth-shape={shape}
      style={{
        position: "absolute",
        left,
        top: top + breatheY,
        width: renderSize,
        height: renderSize,
        backgroundImage: `url(${staticFile(assets.mouth_sheet_path)})`,
        backgroundPosition: `-${x1 * scale}px -${y1 * scale}px`,
        backgroundSize: `${sheetW * scale}px ${sheetH * scale}px`,
        backgroundRepeat: "no-repeat",
        filter: `brightness(${brightness})`,
      }}
    />
  );
};
