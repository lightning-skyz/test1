from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import  QFile
import DCT,DFT,histogram_equalization,gray,nose,buguize,duishu,gamma,test_fenge,test_kuang,test_face3,junzhi
class Stats:
    def __init__(self):
        qufile_stats=QFile('GUI1.ui')
        qufile_stats.open(QFile.ReadOnly)
        qufile_stats.close()
        self.ui = QUiLoader().load(qufile_stats)
        self.ui.runButton.clicked.connect(self.run)

    def path(self):
        from PySide2.QtWidgets import QFileDialog
        filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你的图片",  # 标题
            r"d:\\data",  # 起始目录
            "图片类型 (*.png *.jpg *.bmp)"  # 选择类型过滤项，过滤内容在括号中
        )
        return (filePath)
    def run(self):
        if self.ui.DCTButton.isChecked():
            DCT.DCT1(self.path())
        if self.ui.DFTButton.isChecked():
            DFT.DFT1(self.path())
        if self.ui.zhifangButton.isChecked():
            histogram_equalization.his_eq(self.path())
        if self.ui.noseButton.isChecked():
            nose.addnoise(self.path())
        if self.ui.grayButton.isChecked():
            gray.gray1(self.path())
        if self.ui.buguize.isChecked():
            buguize.buguize(self.path())
        if self.ui.duishu.isChecked():
            buguize.buguize(self.path())
        if self.ui.gamma.isChecked():
            gamma.gamma(self.path())
        if self.ui.junzhi.isChecked():
            junzhi.junzhi(self.path())
        if self.ui.face.isChecked():
            test_face3.face(self.path())
        if self.ui.fenge.isChecked():
            test_fenge.fenge(self.path())
        if self.ui.kuang.isChecked():
            test_kuang.kuang(self.path())
app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()