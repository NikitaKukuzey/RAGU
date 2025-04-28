from dataclasses import dataclass, field


@dataclass
class SearchResult:
    entities: list=field(default_factory=list)
    relations: list=field(default_factory=list)
    summaries: list=field(default_factory=list)
    chunks: list=field(default_factory=list)

    _default_entity_section_title: str = "**Сущности**\nСущность, тип сущности, описание сущности"
    _default_relations_section_title: str = "**Отношения**\nСущность-источник, целевая сущность, описание отношения, ранг отношения"
    _default_summaries_section_title: str = "**Саммари**"
    _default_chunks_section_title: str = "**Тексты**"

    def __str__(self) -> str:
        entity_section = "\n".join(
            [
                f"{entity["entity_name"]}, {entity.get("entity_type")}, {entity.get("description")}"
                for entity in self.entities
            ]
        )

        relations_section = "\n".join(
            [
                f"{relation["source_entity"]}, {relation["target_entity"]}, {relation.get("description")}, {relation.get("rank")}"
                for relation in self.relations
            ]
        )

        summary_section = "\n".join(self.summaries)
        chunks_section = "\n".join(self.chunks)
        return (
            f"{self._default_entity_section_title}\n{entity_section}\n\n" 
            f"{self._default_relations_section_title}\n{relations_section}\n\n"
            f"{self._default_summaries_section_title}\n{summary_section}\n\n" 
            f"{self._default_chunks_section_title}\n{chunks_section}\n\n"
        )