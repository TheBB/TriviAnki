from __future__ import annotations

import contextlib
import json
import re
import shutil
import sqlite3
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

import click
from anki.collection import Collection, ExportAnkiPackageOptions, DeckIdLimit
from anki.media import MediaManager
from anki.models import NotetypeDict
from pydantic import BaseModel, Field, ValidationError, model_validator, model_serializer
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anki.notes import Note as AnkiNote


MODELS: list[str] = [
    "TriviAnki-UG-USA",
    "TriviAnki-UG-Canada",
    "TriviAnki-UG-China",
    "TriviAnki-UG-Australia",
    "TriviAnki-UG-Germany",
    "TriviAnki-UG-Country",
    "TriviAnki-UG-Location",
    "TriviAnki-Art",
    "TriviAnki-SportsTeams",
    "TriviAnki-GreekAlphabet",
    "TriviAnki-Basic",
    "TriviAnki-Basic-Img",
    "TriviAnki-Cloze",
    "TriviAnki-IO",
    "TriviAnki-List",
]


class NoteBase(BaseModel, ABC):
    model_name: str
    guid: str
    mtime: int

    @classmethod
    def load_all(cls, path: Path) -> dict[str, Self]:
        filename = path / f"{cls.__name__}.json"
        if not filename.exists():
            print(f"Warning: notes file for {cls.__name__} not found")
            return {}
        with filename.open("r") as f:
            data = json.load(f)
        return {
            key: cls.model_validate(item) for key, item in data.items()
        }

    @classmethod
    def dump_all(cls, path: Path, notes: dict[str, Self]) -> None:
        filename = path / f"{cls.__name__}.json"
        with filename.open("w") as f:
            json.dump({key: note.model_dump(by_alias=True) for key, note in notes.items()}, f, indent=2)

    @classmethod
    def field_names(cls) -> Iterator[str]:
        for field in cls.model_fields:
            if field not in {"model_name", "guid", "mtime"}:
                yield field

    def files(self) -> Iterator[str]:
        for field in self.field_names():
            value: str = getattr(self, field)
            yield from iter(files_in_str(value))

    def update_anki_note(self, anki_note: AnkiNote) -> None:
        for key, value in self.model_dump(by_alias=True).items():
            if key in {"model_name", "guid", "mtime"}:
                continue
            anki_note[key] = value
        anki_note.guid = self.guid

    @property
    @abstractmethod
    def uid(self) -> str:
        ...


class UsSubdiv(NoteBase):
    model_name: Literal["TriviAnki-UG-USA"]
    name: str = Field(alias="Name")
    map: str = Field(alias="Map")
    flag: str = Field(alias="Flag")
    capital: str = Field(alias="Capital")
    postal_code: str = Field(alias="Postal code")

    @property
    def uid(self) -> str:
        return self.name


class CanadaSubdiv(NoteBase):
    model_name: Literal["TriviAnki-UG-Canada"]
    territory: str = Field(alias="Territory")
    capital: str = Field(alias="Capital")
    map: str = Field(alias="Map")
    extra_map: str = Field(alias="Extra Map")
    blank_map: str = Field(alias="BlankMap")
    flag: str = Field(alias="Flag")
    extra: str = Field(alias="Extra")
    spare: str = Field(alias="Spare")

    @property
    def uid(self) -> str:
        return self.territory


class ChinaSubdiv(NoteBase):
    model_name: Literal["TriviAnki-UG-China"]
    map: str = Field(alias="Map")
    name: str = Field(alias="Name")
    simplified: str = Field(alias="Name in simplified Chinese")
    traditional: str = Field(alias="Name in traditional Chinese")
    pinyin: str = Field(alias="Name in pinyin")
    pronunciation: str = Field(alias="Pronunciation")
    capital: str = Field(alias="Capital")
    etymology: str = Field(alias="Etymology")

    @property
    def uid(self) -> str:
        return self.name


class AustraliaSubdiv(NoteBase):
    model_name: Literal["TriviAnki-UG-Australia"]
    name: str = Field(alias="Name")
    picture: str = Field(alias="Picture")
    capital: str = Field(alias="Capital")
    flag: str = Field(alias="Flag")
    extra: str = Field(alias="Extra")

    @property
    def uid(self) -> str:
        return self.name


class GermanySubdiv(NoteBase):
    model_name: Literal["TriviAnki-UG-Germany"]
    name: str = Field(alias="German State")
    capital: str = Field(alias="Capital")
    map: str = Field(alias="Map")
    coat_of_arms: str = Field(alias="Coat of Arms")
    flag: str = Field(alias="Flag")

    @property
    def uid(self) -> str:
        return self.name


class Country(NoteBase):
    model_name: Literal["TriviAnki-UG-Country"]
    country: str = Field(alias="Country")
    country_info: str = Field(alias="Country info")
    capital: str = Field(alias="Capital")
    capital_info: str = Field(alias="Capital info")
    capital_hint: str = Field(alias="Capital hint")
    flag: str = Field(alias="Flag")
    flag_similarity: str = Field(alias="Flag similarity")
    map: str = Field(alias="Map")

    @property
    def uid(self) -> str:
        return self.country


class Location(NoteBase):
    model_name: Literal["TriviAnki-UG-Location"]
    name: str = Field(alias="Name")
    map: str = Field(alias="Map")
    blank_map: str = Field(alias="BlankMap")

    @property
    def uid(self) -> str:
        return self.name


class Art(NoteBase):
    model_name: Literal["TriviAnki-Art"]
    artwork: str = Field(alias="Artwork")
    artist: str = Field(alias="Artist")
    title: str = Field(alias="Title")
    alternate_title: str = Field(alias="Subtitle/Alternate Titles")
    date: str = Field(alias="Date")
    movement: str = Field(alias="Period/Movement")
    medium: str = Field(alias="Medium")
    nationality: str = Field(alias="Nationality")
    note: str = Field(alias="Note")

    @property
    def uid(self) -> str:
        return self.artwork


class Team(NoteBase):
    model_name: Literal["TriviAnki-SportsTeams"]
    name: str = Field(alias="Team Name")
    city: str = Field(alias="City")
    sport: str = Field(alias="Sport Played")
    logo: str = Field(alias="Logo")
    logo_2: str = Field(alias="Logo 2")
    logo_3: str = Field(alias="Logo 3")
    map: str = Field(alias="Map")
    division: str = Field(alias="Division")
    same_location_teams: str = Field(alias="Same Location Teams")
    spare: str = Field(alias="Spare")

    @property
    def uid(self) -> str:
        return f"{self.city} {self.name}"


class Greek(NoteBase):
    model_name: Literal["TriviAnki-GreekAlphabet"]
    front: str = Field(alias="Front")
    back: str = Field(alias="Back")

    @property
    def uid(self) -> str:
        return self.front


class Basic(NoteBase):
    model_name: Literal["TriviAnki-Basic"]
    front: str = Field(alias="Front")
    back: str = Field(alias="Back")

    @property
    def uid(self) -> str:
        return self.guid


class BasicImg(NoteBase):
    model_name: Literal["TriviAnki-Basic-Img"]
    front: str = Field(alias="Front")
    image: str = Field(alias="Image")
    back: str = Field(alias="Back")

    @property
    def uid(self) -> str:
        return self.guid


class Cloze(NoteBase):
    model_name: Literal["TriviAnki-Cloze"]
    text: str = Field(alias="Text")
    extra: str = Field(alias="Extra")

    @property
    def uid(self) -> str:
        return self.guid


class Occlusion(NoteBase):
    model_name: Literal["TriviAnki-IO"]
    occlusion: str = Field(alias="Occlusion")
    image: str = Field(alias="Image")
    header: str = Field(alias="Header")
    extra: str = Field(alias="Back Extra")
    comments: str = Field(alias="Comments")

    @property
    def uid(self) -> str:
        return self.guid


class List(NoteBase):
    model_name: Literal["TriviAnki-List"]
    header: str = Field(alias="Header")
    item_name: str = Field(alias="ItemName")
    first_context: str = Field(alias="FirstContext")
    items: list[str]
    extra: list[str]

    @model_validator(mode="before")
    @classmethod
    def adapt_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "items" in data:
            return data

        if "Item1" not in data:
            return data

        data["items"] = [data[f"Item{i}"] for i in range(1, 10)]
        data["extra"] = [data[f"Extra{i}"] for i in range(1, 10)]
        return data

    @model_serializer
    def adapt_output(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "guid": self.guid,
            "mtime": self.mtime,
            "Header": self.header,
            "ItemName": self.item_name,
            "FirstContext": self.first_context,
            **{f"Item{i+1}": item for i, item in enumerate(self.items)},
            **{f"Extra{i+1}": item for i, item in enumerate(self.extra)},
        }

    @property
    def uid(self) -> str:
        return self.guid

    def files(self) -> Iterator[str]:
        yield from iter(files_in_str(self.header))
        yield from iter(files_in_str(self.item_name))
        yield from iter(files_in_str(self.first_context))
        for string in chain(self.items, self.extra):
            yield from iter(files_in_str(string))


Note: TypeAlias = (
    UsSubdiv
    | CanadaSubdiv
    | ChinaSubdiv
    | AustraliaSubdiv
    | GermanySubdiv
    | Country
    | Location
    | Art
    | Team
    | Greek
    | Basic
    | BasicImg
    | Cloze
    | Occlusion
    | List
)


class NoteLoader(BaseModel):
    note: Note

    @staticmethod
    def from_anki(note: AnkiNote) -> Note:
        model = note.note_type()
        assert model is not None
        try:
            return NoteLoader.model_validate(
                {
                    "note": {
                        "model_name": model["name"],
                        "guid": note.guid,
                        "mtime": note.mod,
                        **dict(note),
                    },
                }
            ).note
        except ValidationError:
            raise ValueError(f"Unable to load note of type {model['name']}")


class Database(BaseModel):
    root_path: Path

    usa: dict[str, UsSubdiv]
    canada: dict[str, CanadaSubdiv]
    china: dict[str, ChinaSubdiv]
    australia: dict[str, AustraliaSubdiv]
    germany: dict[str, GermanySubdiv]
    countries: dict[str, Country]
    locations: dict[str, Location]
    art: dict[str, Art]
    teams: dict[str, Team]
    greek: dict[str, Greek]

    basic: dict[str, Basic]
    basic_img: dict[str, BasicImg]
    cloze: dict[str, Cloze]
    occlusions: dict[str, Occlusion]
    lists: dict[str, List]

    @property
    def media_path(self) -> Path:
        return self.root_path / "media"

    @property
    def notes_path(self) -> Path:
        return self.root_path / "notes"

    @property
    def models_path(self) -> Path:
        return self.root_path / "models"

    def update_model(self, model: NotetypeDict) -> None:
        filename = self.models_path / f'{model["name"]}.json'
        with filename.open("w") as f:
            json.dump(model, f, indent=4)

    def get_model(self, name: str) -> NotetypeDict:
        filename = self.models_path / f'{name}.json'
        with filename.open("r") as f:
            return json.load(f)

    def insert_note(self, anki_note: AnkiNote) -> None:
        note = NoteLoader.from_anki(anki_note)

        match note:
            case UsSubdiv():
                self.usa[note.uid] = note
            case CanadaSubdiv():
                self.canada[note.uid] = note
            case ChinaSubdiv():
                self.china[note.uid] = note
            case AustraliaSubdiv():
                self.australia[note.uid] = note
            case GermanySubdiv():
                self.germany[note.uid] = note
            case Country():
                self.countries[note.uid] = note
            case Location():
                self.locations[note.uid] = note
            case Art():
                self.art[note.uid] = note
            case Team():
                self.teams[note.uid] = note
            case Greek():
                self.greek[note.uid] = note
            case Basic():
                self.basic[note.uid] = note
            case BasicImg():
                self.basic_img[note.uid] = note
            case Cloze():
                self.cloze[note.uid] = note
            case Occlusion():
                self.occlusions[note.uid] = note
            case List():
                self.lists[note.uid] = note
            case _:
                raise ValueError(f"Unknown note type: {note.model_name}")

        for file in note.files():
            shutil.copy(str(Path(anki_note.col.media._dir) / file), str(self.media_path / file))

    def note_count(self) -> int:
        return (
            len(self.usa)
            + len(self.canada)
            + len(self.china)
            + len(self.australia)
            + len(self.germany)
            + len(self.countries)
            + len(self.locations)
            + len(self.art)
            + len(self.teams)
            + len(self.greek)
            + len(self.basic)
            + len(self.basic_img)
            + len(self.cloze)
            + len(self.occlusions)
            + len(self.lists)
        )

    def notes(self) -> Iterator[Note]:
        notes = chain(
            self.usa.values(),
            self.canada.values(),
            self.china.values(),
            self.australia.values(),
            self.germany.values(),
            self.countries.values(),
            self.locations.values(),
            self.art.values(),
            self.teams.values(),
            self.greek.values(),
            self.basic.values(),
            self.basic_img.values(),
            self.cloze.values(),
            self.occlusions.values(),
            self.lists.values(),
        )
        yield from notes

    @staticmethod
    def load(root_path: Path) -> Database:
        return Database(
            root_path=root_path,
            usa=UsSubdiv.load_all(root_path / "notes"),
            canada=CanadaSubdiv.load_all(root_path / "notes"),
            china=ChinaSubdiv.load_all(root_path / "notes"),
            australia=AustraliaSubdiv.load_all(root_path / "notes"),
            germany=GermanySubdiv.load_all(root_path / "notes"),
            countries=Country.load_all(root_path / "notes"),
            locations=Location.load_all(root_path / "notes"),
            art=Art.load_all(root_path / "notes"),
            teams=Team.load_all(root_path / "notes"),
            greek=Greek.load_all(root_path / "notes"),
            basic=Basic.load_all(root_path / "notes"),
            basic_img=BasicImg.load_all(root_path / "notes"),
            cloze=Cloze.load_all(root_path / "notes"),
            occlusions=Occlusion.load_all(root_path / "notes"),
            lists=List.load_all(root_path / "notes"),
        )

    def dump(self) -> None:
        UsSubdiv.dump_all(self.notes_path, self.usa)
        CanadaSubdiv.dump_all(self.notes_path, self.canada)
        ChinaSubdiv.dump_all(self.notes_path, self.china)
        AustraliaSubdiv.dump_all(self.notes_path, self.australia)
        GermanySubdiv.dump_all(self.notes_path, self.germany)
        Country.dump_all(self.notes_path, self.countries)
        Location.dump_all(self.notes_path, self.locations)
        Art.dump_all(self.notes_path, self.art)
        Team.dump_all(self.notes_path, self.teams)
        Greek.dump_all(self.notes_path, self.greek)
        Basic.dump_all(self.notes_path, self.basic)
        BasicImg.dump_all(self.notes_path, self.basic_img)
        Cloze.dump_all(self.notes_path, self.cloze)
        Occlusion.dump_all(self.notes_path, self.occlusions)
        List.dump_all(self.notes_path, self.lists)

    def stored_media_files(self) -> set[str]:
        filenames: set[str] = set()
        for filename in self.media_path.iterdir():
            filenames.add(filename.name)
        return filenames

    def referenced_media_files(self) -> set[str]:
        filenames: set[str] = set()
        for note in self.notes():
            for filename in note.files():
                filenames.add(filename)
        return filenames

    def clean_media(self) -> None:
        existing_media = self.stored_media_files()
        referenced_media = self.referenced_media_files()
        for filename in existing_media - referenced_media:
            (self.media_path / filename).unlink()
        for filename in referenced_media - existing_media:
            print(f"Warning: missing file: {filename}")

    @contextlib.contextmanager
    @staticmethod
    def enter(root_path: Path) -> Iterator[Database]:
        db = Database.load(root_path)
        yield db
        db.dump()


def files_in_str(string: str, include_remote: bool = False) -> list[str]:
    files = []
    for reg in MediaManager.regexps:
        for match in re.finditer(reg, string):
            fname = match.group("fname")
            is_local = not re.match("(https?|ftp)://", fname.lower())
            if is_local or include_remote:
                files.append(fname)
    return files


def collection_path(path: Path) -> Path:
    if path.is_dir():
        return path / "collection.anki2"
    return path


@contextlib.contextmanager
def collection(path: Path) -> Iterator[Collection]:
    collection = Collection(str(collection_path(path)))
    try:
        yield collection
    finally:
        collection.close()


@click.group
def main() -> None:
    pass


@main.command("import")
@click.option("--deck", "deck_name", help="Name of deck", type=str, default="Trivia")
@click.option("--with-models/--without-models", "with_models", default=False)
@click.argument(
    "collection_path", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path)
)
def import_from_anki(collection_path: Path, deck_name: str, with_models: bool) -> None:
    with collection(collection_path) as col, Database.enter(Path()) as db:
        for noteid in tqdm(col.find_notes(f"deck:{deck_name}"), desc="Processing notes"):
            anki_note = col.get_note(noteid)
            db.insert_note(anki_note)
        db.clean_media()

        if with_models:
            for model_name in MODELS:
                model = col.models.by_name(model_name)
                assert model is not None
                db.update_model(model)


@main.command("clean")
def clean() -> None:
    with Database.enter(Path()) as db:
        db.clean_media()


@main.command("build")
@click.option("--with-media/--without-media", "with_media", default=True)
@click.argument(
    "output_path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path)
)
def build(output_path: Path, with_media: bool) -> None:
    with TemporaryDirectory() as path_str, Database.enter(Path()) as db:
        path = Path(path_str)

        with collection(path) as col:
            # Create models
            for model_id, model_name in enumerate(MODELS):
                model = col.models.new(model_name)
                model.update({**db.get_model(model_name), "id": model["id"]})
                col.models.add_dict(model)

            # Create deck
            deck = col.decks.new_deck()
            deck.name = "Trivia"
            col.decks.add_deck(deck)
            deck_id = col.decks.id_for_name("Trivia")
            assert deck_id is not None

            # Create notes
            for note in tqdm(db.notes(), desc="Processing notes", total=db.note_count()):
                model = col.models.by_name(note.model_name)
                assert model is not None
                anki_note = col.new_note(model)
                note.update_anki_note(anki_note)
                col.add_note(anki_note, deck_id)

            # Add media
            if with_media:
                for filename in tqdm(db.stored_media_files(), desc="Processing media files"):
                    col.media.add_file(str(db.media_path / filename))

        # Tweak modified times
        connection = sqlite3.connect(str(collection_path(path)))
        for note in tqdm(db.notes(), desc="Updating modification timestamps", total=db.note_count()):
            connection.execute("UPDATE notes SET mod = ? WHERE guid = ?", (note.mtime, note.guid))
        connection.commit()
        connection.close()

        shutil.copy(str(collection_path(path)), "test.col")

        with collection(Path(path_str)) as col:
            # Check database integrity
            err, ok = col.fix_integrity()
            assert ok, err

            # Export
            col.export_anki_package(
                out_path=str(output_path.absolute()),
                options=ExportAnkiPackageOptions(
                    with_scheduling=False,
                    with_deck_configs=False,
                    with_media=True,
                    legacy=False,
                ),
                limit=DeckIdLimit(deck_id),
            )
