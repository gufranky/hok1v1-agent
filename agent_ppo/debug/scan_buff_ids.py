#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
from pathlib import Path


def main():
    frames_dir = Path(__file__).resolve().parent / "frames"
    skill_ids = set()
    mark_max_layers = {}
    for path in sorted(frames_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as file:
            observation = json.load(file)
        for agent_obs in observation.values():
            frame_state = agent_obs.get("frame_state", {})
            for hero in frame_state.get("hero_states", []):
                actor_state = hero.get("actor_state", hero)
                buff_state = actor_state.get("buff_state", {})
                for item in buff_state.get("buff_skills", []):
                    skill_ids.add(item.get("configId", 0))
                for item in buff_state.get("buff_marks", []):
                    mark_id = item.get("configId", 0)
                    layer = item.get("layer", 0)
                    mark_max_layers[mark_id] = max(mark_max_layers.get(mark_id, 0), layer)

    print("buff_skills:")
    for skill_id in sorted(skill_ids):
        print(skill_id)
    print("\nbuff_marks:")
    for mark_id in sorted(mark_max_layers):
        print(f"{mark_id}: {mark_max_layers[mark_id]}")


if __name__ == "__main__":
    main()
