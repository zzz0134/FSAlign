#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[;\/\[\]\"{}\(\)=\+\\_\-><@`,\?!]")
_PERIOD_RE = re.compile(r"(?!<=\d)\.(?!\d)")
_COMMA_NUM_RE = re.compile(r"(\d),(\d)")

_NUM_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def normalize_vqa_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = _COMMA_NUM_RE.sub(r"\1\2", text)
    text = _PERIOD_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    words: List[str] = []
    for token in _SPACE_RE.sub(" ", text).split(" "):
        token = token.strip()
        if not token or token in _ARTICLES:
            continue
        words.append(_NUM_MAP.get(token, token))
    return " ".join(words)


def normalize_question_text(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text).strip())


def select_canonical_answer(answers: Sequence[str], multiple_choice_answer: str = "") -> str:
    normalized = [normalize_vqa_answer(a) for a in answers]
    normalized = [a for a in normalized if a]
    if not normalized and multiple_choice_answer:
        normalized = [normalize_vqa_answer(multiple_choice_answer)]
    if not normalized:
        return ""

    counts: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}
    for idx, ans in enumerate(normalized):
        counts[ans] = counts.get(ans, 0) + 1
        first_pos.setdefault(ans, idx)
    best = sorted(counts.keys(), key=lambda a: (-counts[a], first_pos[a], a))[0]
    return best


def format_question_prompt(question: str, template: str) -> str:
    return template.format(q=normalize_question_text(question))


def format_answer_prompt(answer: str, template: str) -> str:
    return template.format(a=normalize_vqa_answer(answer))


def fuse_query_features(image_feats: torch.Tensor, question_feats: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if image_feats.shape != question_feats.shape:
        raise ValueError(f"Shape mismatch for fusion: {tuple(image_feats.shape)} vs {tuple(question_feats.shape)}")
    if mode == "mean":
        fused = 0.5 * (image_feats + question_feats)
    elif mode == "sum":
        fused = image_feats + question_feats
    else:
        raise ValueError(f"Unsupported VQA fusion mode: {mode}")
    return l2norm(fused)


def vqa_soft_accuracy(pred_answer: str, gt_answers: Sequence[str]) -> float:
    pred = normalize_vqa_answer(pred_answer)
    if not pred:
        return 0.0
    matches = sum(1 for ans in gt_answers if normalize_vqa_answer(ans) == pred)
    return min(1.0, float(matches) / 3.0)


def vqa_topk_scores(
    topk_indices: torch.Tensor,
    idx_to_answer: Sequence[str],
    gt_answers_batch: Sequence[Sequence[str]],
) -> Tuple[float, float]:
    total_top1 = 0.0
    total_topk = 0.0
    rows = topk_indices.detach().cpu().tolist()
    for pred_ids, gt_answers in zip(rows, gt_answers_batch):
        total_top1 += vqa_soft_accuracy(idx_to_answer[pred_ids[0]], gt_answers)
        total_topk += max(vqa_soft_accuracy(idx_to_answer[j], gt_answers) for j in pred_ids)
    return total_top1, total_topk


def _load_vqav2_json(root: Path, split: str) -> Tuple[dict, dict]:
    q_path = root / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    a_path = root / f"v2_mscoco_{split}2014_annotations.json"
    assert q_path.exists(), f"VQAv2 questions json not found: {q_path}"
    assert a_path.exists(), f"VQAv2 annotations json not found: {a_path}"
    questions = json.loads(q_path.read_text(encoding="utf-8"))
    annotations = json.loads(a_path.read_text(encoding="utf-8"))
    return questions, annotations


def build_vqav2_answer_vocab(vqav2_root: str, top_k: int) -> Tuple[List[str], Dict[str, int]]:
    root = Path(vqav2_root)
    _, ann_json = _load_vqav2_json(root, "train")
    counter: Counter = Counter()
    for ann in ann_json["annotations"]:
        for ans in ann.get("answers", []):
            norm = normalize_vqa_answer(ans.get("answer", ""))
            if norm:
                counter[norm] += 1
    vocab = [ans for ans, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
    answer_to_idx = {ans: idx for idx, ans in enumerate(vocab)}
    return vocab, answer_to_idx


def build_vqa_soft_target(
    answers: Sequence[str],
    answer_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[List[int], List[float]]:
    counts: Counter = Counter()
    for ans in answers:
        norm = normalize_vqa_answer(ans)
        if norm:
            counts[norm] += 1

    if not counts or answer_to_idx is None:
        return [], []

    entries: List[Tuple[int, float]] = []
    for ans, count in counts.items():
        idx = int(answer_to_idx.get(ans, -1))
        if idx < 0:
            continue
        score = min(1.0, float(count) / 3.0)
        if score > 0.0:
            entries.append((idx, score))

    entries.sort(key=lambda x: x[0])
    return [idx for idx, _ in entries], [score for _, score in entries]


def sparse_vqa_targets_to_dense(
    target_indices_batch: Sequence[Sequence[int]],
    target_scores_batch: Sequence[Sequence[float]],
    num_answers: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    target = torch.zeros(len(target_indices_batch), num_answers, device=device, dtype=dtype)
    for row, (indices, scores) in enumerate(zip(target_indices_batch, target_scores_batch)):
        if not indices:
            continue
        idx_t = torch.tensor(indices, device=device, dtype=torch.long)
        score_t = torch.tensor(scores, device=device, dtype=dtype)
        target[row, idx_t] = score_t
    return target


def sparse_vqa_targets_to_embeddings(
    answer_feats: torch.Tensor,
    target_indices_batch: Sequence[Sequence[int]],
    target_scores_batch: Sequence[Sequence[float]],
    fallback_answer_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for row_idx, (indices, scores) in enumerate(zip(target_indices_batch, target_scores_batch)):
        if indices:
            idx_t = torch.tensor(indices, device=answer_feats.device, dtype=torch.long)
            score_t = torch.tensor(scores, device=answer_feats.device, dtype=answer_feats.dtype)
            vec = (answer_feats.index_select(0, idx_t) * score_t.unsqueeze(1)).sum(dim=0)
        elif fallback_answer_feats is not None:
            vec = fallback_answer_feats[row_idx].to(answer_feats.device)
        else:
            vec = torch.zeros(answer_feats.size(1), device=answer_feats.device, dtype=answer_feats.dtype)
        rows.append(vec)
    if not rows:
        return torch.empty(0, answer_feats.size(1), device=answer_feats.device, dtype=answer_feats.dtype)
    return l2norm(torch.stack(rows, dim=0))


class VQAv2ClassificationDataset(Dataset):
    def __init__(
        self,
        vqav2_root: str,
        split: str,
        answer_to_idx: Optional[Dict[str, int]] = None,
        drop_oov: bool = False,
        max_items: Optional[int] = None,
    ):
        super().__init__()
        assert split in {"train", "val"}, f"Unsupported split: {split}"
        self.root = Path(vqav2_root)
        self.split = split

        questions_json, annotations_json = _load_vqav2_json(self.root, split)
        img_dir = self.root / f"{split}2014"
        assert img_dir.exists(), f"VQAv2 image dir not found: {img_dir}"

        ann_by_qid = {int(a["question_id"]): a for a in annotations_json["annotations"]}
        items = []
        labeled = 0
        missing_images = 0
        covered_soft_mass = 0.0
        total_soft_mass = 0.0

        for q in questions_json["questions"]:
            qid = int(q["question_id"])
            ann = ann_by_qid.get(qid)
            if ann is None:
                continue

            raw_answers = [a.get("answer", "") for a in ann.get("answers", [])]
            gt_answers = [normalize_vqa_answer(a) for a in raw_answers]
            gt_answers = [a for a in gt_answers if a]
            canonical = select_canonical_answer(raw_answers, ann.get("multiple_choice_answer", ""))
            if not canonical:
                continue

            label = -1
            soft_target_indices: List[int] = []
            soft_target_scores: List[float] = []
            if answer_to_idx is not None:
                label = int(answer_to_idx.get(canonical, -1))
                soft_target_indices, soft_target_scores = build_vqa_soft_target(raw_answers, answer_to_idx)
                if drop_oov and not soft_target_indices:
                    continue

            image_id = int(q["image_id"])
            img_path = img_dir / f"COCO_{split}2014_{image_id:012d}.jpg"
            if not img_path.exists():
                missing_images += 1
                continue

            if soft_target_indices or answer_to_idx is None:
                labeled += 1
            if gt_answers:
                total_soft_mass += sum(min(1.0, float(count) / 3.0) for count in Counter(gt_answers).values())
                covered_soft_mass += sum(soft_target_scores)

            items.append(
                {
                    "image_path": str(img_path),
                    "question": str(q["question"]),
                    "label": label,
                    "canonical_answer": canonical,
                    "answers": gt_answers,
                    "question_id": qid,
                    "image_id": image_id,
                    "soft_target_indices": soft_target_indices,
                    "soft_target_scores": soft_target_scores,
                }
            )
            if max_items is not None and len(items) >= max_items:
                break

        if not items:
            raise AssertionError(f"No VQAv2 items built for split='{split}' under {self.root}")

        self.items = items
        self.answer_coverage = float(labeled) / float(len(items))
        self.answer_mass_coverage = float(covered_soft_mass) / float(total_soft_mass) if total_soft_mass > 0.0 else 0.0
        self.missing_images = missing_images

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        return (
            img,
            item["question"],
            int(item["label"]),
            item["canonical_answer"],
            item["answers"],
            int(item["question_id"]),
            list(item["soft_target_indices"]),
            list(item["soft_target_scores"]),
        )


def collate_vqa(batch):
    images = [b[0] for b in batch]
    questions = [str(b[1]) for b in batch]
    labels = torch.tensor([int(b[2]) for b in batch], dtype=torch.long)
    canonical_answers = [str(b[3]) for b in batch]
    answers = [list(b[4]) for b in batch]
    question_ids = [int(b[5]) for b in batch]
    soft_target_indices = [list(b[6]) for b in batch]
    soft_target_scores = [list(b[7]) for b in batch]
    return images, questions, labels, canonical_answers, answers, question_ids, soft_target_indices, soft_target_scores
