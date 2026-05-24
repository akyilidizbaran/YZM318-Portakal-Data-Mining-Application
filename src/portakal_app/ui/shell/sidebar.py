from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QSizePolicy, QPushButton, QFrame, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from portakal_app.models import CategoryDefinition
from portakal_app.ui import i18n
from portakal_app.ui.icons import get_widget_icon


class SidebarCategoryList(QFrame):
    categorySelected = Signal(str)
    workflowInfoRequested = Signal()

    def __init__(self, categories: list[CategoryDefinition], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("categoryPanel")
        self._categories = categories
        self._items_by_category: dict[str, QListWidgetItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 10)
        layout.setSpacing(4)

        self._list = QListWidget()
        self._list.setObjectName("categoryList")
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._list.setIconSize(QSize(20, 20))
        self._list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._list.currentItemChanged.connect(self._emit_selected_category)
        layout.addWidget(self._list)
        self._populate_list()

        layout.addStretch(1)
        self._workflow_info_button = QPushButton(i18n.t("Workflow Info"))
        self._workflow_info_button.setObjectName("workflowInfoButton")
        self._workflow_info_button.clicked.connect(self.workflowInfoRequested.emit)
        layout.addWidget(self._workflow_info_button)

        self._sync_list_height()

    def _populate_list(self, current_category_id: str | None = None) -> None:
        current_category_id = current_category_id or self.current_category_id()
        self._list.clear()
        self._items_by_category.clear()
        for category in self._categories:
            item = QListWidgetItem(category.label)
            item.setData(Qt.ItemDataRole.UserRole, category.id)
            if category.icon_name:
                item.setIcon(get_widget_icon(category.icon_name))
            self._list.addItem(item)
            self._items_by_category[category.id] = item
        if current_category_id is not None:
            self.set_current_category(current_category_id)
        self._sync_list_height()

    def set_categories(self, categories: list[CategoryDefinition]) -> None:
        self._categories = categories
        self._populate_list()

    def refresh_translations(self) -> None:
        self._workflow_info_button.setText(i18n.t("Workflow Info"))
        for category in self._categories:
            item = self._items_by_category.get(category.id)
            if item is not None:
                item.setText(category.label)

    def _emit_selected_category(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        category_id = current.data(Qt.ItemDataRole.UserRole)
        self.categorySelected.emit(category_id)

    def set_current_category(self, category_id: str) -> None:
        item = self._items_by_category.get(category_id)
        if item is not None:
            self._list.setCurrentItem(item)

    def current_category_id(self) -> str | None:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def recommended_width(self) -> int:
        metrics = QFontMetrics(self._list.font())
        longest_label = max((category.label for category in self._categories), key=len, default="")
        text_width = metrics.horizontalAdvance(longest_label)
        layout_margins = self.layout().contentsMargins()
        frame_width = layout_margins.left() + layout_margins.right()
        item_padding = 92
        return max(188, text_width + frame_width + item_padding)

    def _sync_list_height(self) -> None:
        if self._list.count() == 0:
            return
        row_height = self._list.sizeHintForRow(0)
        spacing = 2
        frame = self._list.frameWidth() * 2
        viewport_margins = 8
        total = (row_height * self._list.count()) + (spacing * (self._list.count() - 1)) + frame + viewport_margins
        self._list.setFixedHeight(total)
