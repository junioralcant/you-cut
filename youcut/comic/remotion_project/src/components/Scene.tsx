// Scene — wrapper que aplica Ken Burns + transitionIn (crossfade/cut/wipe)
// e instancia Characters da cena.
//
// O Sequence em ComicVideo já cuida do offset temporal; o frame relativo
// dentro da Scene é obtido via `useCurrentFrame()` (Remotion ajusta).

import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type { CharacterAssets, RemotionScene } from "../types";
import { Character } from "./Character";
import { Shake } from "./Shake";

const TRANSITION_DURATION_SEC = 0.3;

interface SceneProps {
  scene: RemotionScene;
  characters: Record<string, CharacterAssets>;
  blinkPeriodSec?: number;
}

export const Scene: React.FC<SceneProps> = ({
  scene,
  characters,
  blinkPeriodSec,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const sceneDurationSec = scene.end_sec - scene.start_sec;
  const sceneDurationFrames = Math.max(1, Math.round(sceneDurationSec * fps));
  const transitionFrames = Math.max(1, Math.round(TRANSITION_DURATION_SEC * fps));

  // Ken Burns: scale + translate suave ao longo da cena.
  const scaleFrom = scene.ken_burns?.scale_from ?? 1.0;
  const scaleTo = scene.ken_burns?.scale_to ?? 1.0;
  const fromXY = scene.ken_burns?.from ?? [0, 0];
  const toXY = scene.ken_burns?.to ?? [0, 0];
  const scale = interpolate(frame, [0, sceneDurationFrames], [scaleFrom, scaleTo], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const tx = interpolate(frame, [0, sceneDurationFrames], [fromXY[0], toXY[0]], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const ty = interpolate(frame, [0, sceneDurationFrames], [fromXY[1], toXY[1]], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Transition in
  let opacity = 1;
  let wipeProgress = 1;
  if (scene.transition_in === "crossfade") {
    opacity = interpolate(frame, [0, transitionFrames], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  } else if (scene.transition_in === "wipe") {
    wipeProgress = interpolate(frame, [0, transitionFrames], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }

  const clipPath =
    scene.transition_in === "wipe"
      ? `inset(0 ${(1 - wipeProgress) * 100}% 0 0)`
      : undefined;

  // Filtra lipSync para os personagens da cena.
  const characterIds = scene.character_ids ?? [];

  return (
    <AbsoluteFill
      style={{
        opacity,
        clipPath,
        transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
        transformOrigin: "center center",
      }}
      data-scene-index={scene.index}
      data-time-sec={t.toFixed(3)}
    >
      <Shake shakes={scene.shakes ?? []}>
        {characterIds.map((charId) => {
          const assets = characters[charId];
          if (!assets) return null;
          return (
            <Character
              key={charId}
              characterId={charId}
              assets={assets}
              lipSync={scene.lip_sync ?? []}
              blinkPeriodSec={blinkPeriodSec}
            />
          );
        })}
      </Shake>
    </AbsoluteFill>
  );
};
