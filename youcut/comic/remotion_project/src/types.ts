// Espelha 1:1 o modelo Pydantic `RemotionInputProps` em `youcut/models.py`.
// Quaisquer mudanças em campos devem ser sincronizadas em ambos os lados.

export type MouthShape =
  | "closed"
  | "open_mid"
  | "open_wide"
  | "open_round";

export type TransitionType = "cut" | "crossfade" | "wipe";

// `MouthEvent` e `Shake` usam tempos RELATIVOS ao início da Scene em
// que ocorrem. O orquestrador Python deve subtrair `scene.start_sec`
// antes de emitir o JSON. Dentro do Remotion, `useCurrentFrame()` é
// relativo ao `<Sequence>` que envolve a Scene.

export interface MouthEvent {
  character_id: string;
  start_sec: number;
  end_sec: number;
  shape: MouthShape;
}

export interface KenBurns {
  scale_from?: number;
  scale_to?: number;
  from?: [number, number];
  to?: [number, number];
}

export interface Shake {
  at_sec: number;
  intensity: number;
}

export interface RemotionScene {
  index: number;
  start_sec: number;
  end_sec: number;
  character_ids: string[];
  speaker_id: string | null;
  ken_burns: KenBurns;
  transition_in: TransitionType;
  shakes: Shake[];
  lip_sync: MouthEvent[];
}

export interface CharacterAssets {
  anchor_path?: string;
  mouth_sheet_path?: string;
  cells?: Record<MouthShape, [number, number, number, number]>;
}

export interface InputProps {
  audio_path: string;
  duration_sec: number;
  fps?: number;
  width?: number;
  height?: number;
  characters: Record<string, CharacterAssets>;
  scenes: RemotionScene[];
  background_color?: string;
}
