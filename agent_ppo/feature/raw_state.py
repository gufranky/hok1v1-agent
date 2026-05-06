#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from dataclasses import dataclass, field
from typing import Dict, List, Optional


def camp_to_int(camp):
    if isinstance(camp, int):
        if camp in (0, 1):
            return camp
        return camp - 1
    if isinstance(camp, str) and camp.startswith("PLAYERCAMP_"):
        tail = camp.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail) - 1
    return -1


def subtype_eq(value, expected_int, expected_name):
    return value == expected_int or value == expected_name


def actor_type_eq(value, expected_name):
    return value == expected_name


@dataclass
class SkillSlot:
    slot_type: str
    level: int
    usable: bool
    cooldown: int
    cooldown_max: int


@dataclass
class BuffInfo:
    skill_ids: List[int] = field(default_factory=list)
    mark_ids: List[int] = field(default_factory=list)
    mark_layers: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, buff_state: Dict):
        buff_state = buff_state or {}
        skill_ids = [item.get("configId", 0) for item in buff_state.get("buff_skills", [])]
        mark_ids = []
        mark_layers = []
        for item in buff_state.get("buff_marks", []):
            mark_ids.append(item.get("configId", 0))
            mark_layers.append(item.get("layer", 0))
        return cls(skill_ids=skill_ids, mark_ids=mark_ids, mark_layers=mark_layers)


@dataclass
class Unit:
    config_id: int
    runtime_id: int
    camp: int
    actor_type: str
    sub_type: str
    behave: str
    location: List[int]
    forward: List[int]
    hp: int
    max_hp: int
    ep: int
    max_ep: int
    attack_range: int
    attack_target: int
    buff: BuffInfo

    @classmethod
    def from_actor_state(cls, actor_state: Dict):
        values = actor_state.get("values", {})
        return cls(
            config_id=actor_state.get("config_id", 0),
            runtime_id=actor_state.get("runtime_id", 0),
            camp=camp_to_int(actor_state.get("camp", -1)),
            actor_type=actor_state.get("actor_type", ""),
            sub_type=actor_state.get("sub_type", ""),
            behave=actor_state.get("behav_mode", ""),
            location=[actor_state.get("location", {}).get("x", 0), actor_state.get("location", {}).get("z", 0)],
            forward=[actor_state.get("forward", {}).get("x", 0), actor_state.get("forward", {}).get("z", 0)],
            hp=actor_state.get("hp", 0),
            max_hp=actor_state.get("max_hp", 1),
            ep=values.get("ep", actor_state.get("ep", 0)),
            max_ep=values.get("max_ep", actor_state.get("max_ep", 1)),
            attack_range=actor_state.get("attack_range", 0),
            attack_target=actor_state.get("attack_target", 0),
            buff=BuffInfo.from_dict(actor_state.get("buff_state")),
        )


@dataclass
class Hero:
    player_id: int
    unit: Unit
    level: int
    exp: int
    money: int
    money_total: int
    kill_cnt: int
    dead_cnt: int
    assist_cnt: int
    is_in_grass: bool
    skills: Dict[str, SkillSlot]

    @classmethod
    def from_dict(cls, hero_state: Dict):
        slot_states = hero_state.get("skill_state", {}).get("slot_states", [])
        skill_map: Dict[str, SkillSlot] = {}
        for item in slot_states:
            slot = SkillSlot(
                slot_type=item.get("slot_type", ""),
                level=item.get("level", 0),
                usable=item.get("usable", False),
                cooldown=item.get("cooldown", 0),
                cooldown_max=max(item.get("cooldown_max", 1), 1),
            )
            skill_map[slot.slot_type] = slot
        return cls(
            player_id=hero_state.get("player_id", 0),
            unit=Unit.from_actor_state(hero_state.get("actor_state", {})),
            level=hero_state.get("level", 1),
            exp=hero_state.get("exp", 0),
            money=hero_state.get("money", 0),
            money_total=hero_state.get("moneyCnt", 0),
            kill_cnt=hero_state.get("killCnt", 0),
            dead_cnt=hero_state.get("deadCnt", 0),
            assist_cnt=hero_state.get("assistCnt", 0),
            is_in_grass=hero_state.get("isInGrass", False),
            skills=skill_map,
        )


@dataclass
class Bullet:
    runtime_id: int
    camp: int
    source_actor: int
    slot_type: str
    location: List[int]

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            runtime_id=data.get("runtime_id", 0),
            camp=camp_to_int(data.get("camp", -1)),
            source_actor=data.get("source_actor", 0),
            slot_type=data.get("slot_type", ""),
            location=[data.get("location", {}).get("x", 0), data.get("location", {}).get("z", 0)],
        )


@dataclass
class Cake:
    camp: int
    location: List[int]


@dataclass
class DeadEvent:
    killer_runtime_id: int
    death_runtime_id: int
    death_sub_type: str

    @classmethod
    def from_dict(cls, data: Dict):
        killer = data.get("killer", {})
        death = data.get("death", {})
        return cls(
            killer_runtime_id=killer.get("runtime_id", 0),
            death_runtime_id=death.get("runtime_id", 0),
            death_sub_type=death.get("sub_type", ""),
        )


@dataclass
class ParsedState:
    player_id: int
    player_camp: int
    frame_no: int
    heroes: List[Hero]
    npc_units: List[Unit]
    bullets: List[Bullet]
    cakes: List[Cake]
    dead_events: List[DeadEvent]
    my_hero: Hero
    enemy_hero: Hero
    my_tower: Optional[Unit]
    enemy_tower: Optional[Unit]
    my_soldiers: List[Unit]
    enemy_soldiers: List[Unit]
    id_to_unit_type: Dict[int, str]


def build_parsed_state(observation: Dict) -> ParsedState:
    frame_state = observation["frame_state"]
    player_id = observation["player_id"]
    player_camp = camp_to_int(observation["player_camp"])

    heroes = [Hero.from_dict(item) for item in frame_state.get("hero_states", [])]
    my_hero = next(hero for hero in heroes if hero.player_id == player_id)
    enemy_hero = next(hero for hero in heroes if hero.player_id != player_id)

    npc_units = [Unit.from_actor_state(item) for item in frame_state.get("npc_states", [])]
    bullets = [Bullet.from_dict(item) for item in frame_state.get("bullets", [])]
    cakes: List[Cake] = []
    for item in frame_state.get("cakes", []) or []:
        loc = item.get("collider", {}).get("location", {})
        x = loc.get("x", 0)
        z = loc.get("z", 0)
        cakes.append(Cake(camp=int(x > 0), location=[x, z]))

    dead_events = [DeadEvent.from_dict(item) for item in frame_state.get("frame_action", {}).get("dead_action", [])]

    my_tower = None
    enemy_tower = None
    my_soldiers: List[Unit] = []
    enemy_soldiers: List[Unit] = []
    id_to_unit_type: Dict[int, str] = {
        my_hero.unit.runtime_id: "hero",
        enemy_hero.unit.runtime_id: "hero",
    }

    for unit in npc_units:
        if actor_type_eq(unit.actor_type, "ACTOR_MONSTER") and subtype_eq(unit.sub_type, 0, "ACTOR_SUB_SOLDIER"):
            id_to_unit_type[unit.runtime_id] = "soldier"
            if unit.camp == player_camp:
                my_soldiers.append(unit)
            elif unit.camp == 1 - player_camp:
                enemy_soldiers.append(unit)
        elif actor_type_eq(unit.actor_type, "ACTOR_ORGAN"):
            id_to_unit_type[unit.runtime_id] = "organ"
            if subtype_eq(unit.sub_type, 21, "ACTOR_SUB_TOWER"):
                if unit.camp == player_camp:
                    my_tower = unit
                elif unit.camp == 1 - player_camp:
                    enemy_tower = unit

    my_soldiers.sort(key=lambda item: distance(item.location, my_hero.unit.location))
    enemy_soldiers.sort(key=lambda item: distance(item.location, my_hero.unit.location))

    return ParsedState(
        player_id=player_id,
        player_camp=player_camp,
        frame_no=frame_state.get("frame_no", frame_state.get("frameNo", 0)),
        heroes=heroes,
        npc_units=npc_units,
        bullets=bullets,
        cakes=cakes,
        dead_events=dead_events,
        my_hero=my_hero,
        enemy_hero=enemy_hero,
        my_tower=my_tower,
        enemy_tower=enemy_tower,
        my_soldiers=my_soldiers,
        enemy_soldiers=enemy_soldiers,
        id_to_unit_type=id_to_unit_type,
    )


def distance(a: List[int], b: List[int]) -> float:
    dx = a[0] - b[0]
    dz = a[1] - b[1]
    return (dx * dx + dz * dz) ** 0.5
