import os
import sys
import json
from PyQt5 import QtCore, QtWidgets, uic
from utils import Utils

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
config_path = 'gui/config.json'
config = json.loads(open(config_path, 'rb').read())
utils = Utils(config)
Ui_MainWindow, QtBaseClass = uic.loadUiType(config['ui'])


class MainUI(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self._bind_ui_()
        self._init_ui_()

    def _bind_ui_(self):
        self.selectBtn.clicked.connect(self.openfile)
        self.startSearch.clicked.connect(self.start_search)
        self.startSearchDuplicate.clicked.connect(self.start_search_duplicate)
        self.resultTable.doubleClicked.connect(self.double_click_search_table)
        self.resultTableDuplicate.doubleClicked.connect(
            self.double_click_duplicate_table)
        self.addSearchDir.clicked.connect(self.add_search_dir)
        self.updateIndex.clicked.connect(self.sync_index)
        self.removeInvalidIndex.clicked.connect(self.remove_invalid_index)

    def _init_ui_(self):
        if os.path.exists(utils.exists_index_path):
            self.exists_index = utils.get_exists_index()  # 加载索引
        self.resultTable.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)  # 填充显示表格
        self.resultTable.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)  # 表格设置只读
        self.resultTableDuplicate.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        self.resultTableDuplicate.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)
        self.resultTableDuplicate.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.resultTableDuplicate.setSortingEnabled(True)
        self.searchDirTable.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.searchDirTable.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.update_dir_table()

    def openfile(self):
        self.input_path = QtWidgets.QFileDialog.getOpenFileName(
            self, '选择图片', '', 'Image files(*.*)')
        self.filePath.setText(self.input_path[0])
        self.filePath.setToolTip(f'<img width=300 src="{self.input_path[0]}">')

    def double_click_search_table(self, info):
        file_path = os.path.normpath(
            self.resultTable.item(info.row(), 0).text())
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            QtWidgets.QMessageBox.warning(self, '警告', '图片文件不存在：' + file_path)

    def double_click_duplicate_table(self, info):
        col = info.column()
        if col > 1:
            return
        row = info.row()
        file_path = self.resultTableDuplicate.item(row, col).text()
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            QtWidgets.QMessageBox.warning(self, '警告', '图片文件不存在：' + file_path)

    def start_search(self):
        if not hasattr(self, 'input_path'):
            self.openfile()
        if (config['search_dir']
                == []) or (not os.path.exists(utils.exists_index_path)):
            QtWidgets.QMessageBox.information(self, '提示', '索引都没有建搜你🐎 搜')
            return
        self.resultTable.setRowCount(0)  # 清空表格
        nc = self.resultCount.value()
        nc = nc if nc <= len(self.exists_index) else len(self.exists_index)
        results = utils.checkout(self.input_path[0], self.exists_index, nc)
        for i in results:
            row = self.resultTable.rowCount()
            self.resultTable.insertRow(row)
            item_sim = QtWidgets.QTableWidgetItem(f'{i[0]:.2f} %')
            item_sim.setTextAlignment(QtCore.Qt.AlignHCenter
                                      | QtCore.Qt.AlignVCenter)
            item_path = QtWidgets.QTableWidgetItem(i[1])
            item_path.setToolTip(f'{i[1]}<br><img width=300 src="{i[1]}">')
            self.resultTable.setItem(row, 0, item_path)
            self.resultTable.setItem(row, 1, item_sim)

    def start_search_duplicate(self):
        if (config['search_dir']
                == []) or (not os.path.exists(utils.exists_index_path)):
            QtWidgets.QMessageBox.information(self, '提示', '索引都没有建查你🐎 查')
            return
        self.resultTableDuplicate.setRowCount(0)  # 清空表格
        threshold = self.similarityThreshold.value()
        same_folder = self.sameFolder.isChecked()
        for i in utils.get_duplicate(self.exists_index, threshold,
                                     same_folder):
            row = self.resultTableDuplicate.rowCount()
            self.resultTableDuplicate.insertRow(row)
            item_path_a = QtWidgets.QTableWidgetItem(i[0])
            item_path_a.setToolTip(f'{i[0]}<br><img width=300 src="{i[0]}">')
            item_path_b = QtWidgets.QTableWidgetItem(i[1])
            item_path_b.setToolTip(f'{i[1]}<br><img width=300 src="{i[1]}">')
            item_sim = QtWidgets.QTableWidgetItem(f'{i[2]:.2f} %')
            item_sim.setTextAlignment(QtCore.Qt.AlignHCenter
                                      | QtCore.Qt.AlignVCenter)
            self.resultTableDuplicate.setItem(row, 0, item_path_a)
            self.resultTableDuplicate.setItem(row, 1, item_path_b)
            self.resultTableDuplicate.setItem(row, 2, item_sim)

    def update_dir_table(self):
        self.searchDirTable.setRowCount(0)
        for i in config['search_dir']:
            row = self.searchDirTable.rowCount()
            self.searchDirTable.insertRow(row)
            item = QtWidgets.QTableWidgetItem(i)
            self.searchDirTable.setItem(row, 0, item)

    def add_search_dir(self):
        self.input_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, '选择一个需要索引的图片目录')
        if not self.input_path:
            return
        config['search_dir'].append(self.input_path)
        self.save_settings()
        self.update_dir_table()

    def remove_invalid_index(self):
        utils.remove_nonexists()
        self.exists_index = utils.get_exists_index()
        QtWidgets.QMessageBox.information(self, '提示', '无效索引已删除')

    def sync_index(self):
        utils.remove_nonexists()
        for image_dir in config['search_dir']:
            res = utils.get_need_index(image_dir)
            need_index, exists_index, metainfo = res
            utils.update_ir_index(need_index)
            utils.save_meta_files(exists_index, metainfo)
        self.exists_index = utils.get_exists_index()
        QtWidgets.QMessageBox.information(self, '提示', '索引同步已完成')

    def save_settings(self):
        with open(config_path, 'wb') as wp:
            wp.write(utils.dumps(
                config,
                indent=2,
            ).encode('UTF-8'))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec_())
