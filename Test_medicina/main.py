from ReadWriteFiles import ReadWriteFiles as RWFiles
from ConvertDan import ConverDan as CD
import copy
from Models import NWModel as NWM


# модель нейронки
def model_0(self, input_dan):
        x = self.Dense_drop(input_dan, 6760, "softsign", 0.12)
        return self.Dense_drop(x, 700, "softsign", 0.0)

if __name__ == "__main__":
    bolezn1="Здравствуйте, у меня тяжесть в правом подреберье \n \
            и ощущение дискомфорта и вообще боль в ребрах справа, \n  \
            также ощущение горечи во рту, дискомфорт в животе,утренняя тошнота. \n  \
            Еще я заметил сосудистые звездочки на коже и общее недомогание."

    bolezn2 = "У моего малыша упала температуры тела ниже 36 °С и стул белый. \n  \
                У него сонливость (может спать по несколько часов без пробуждения на прием пищи). \n  \
                Склеры глаз ребенка становятся все более желтыми, желтый окрас более трех недель. \n  \
                У Сони вялое сосание, окрашиванию кожи и слизистых в желтый цвет, \n  \
                заторможенность и переодическая рвота."

    _rwfiles = RWFiles("O:/Python/TestDima/TestDima/Болезни")
    _cd = CD()
# Формирование данных, предварительный анализ нейронки было в  main_Genetika.py

# Установка переменных
    maxWordsCount, xLen, step = 20000, 50, 10
    _dan =[maxWordsCount, xLen, step]

    _nwm = NWM(_cd.Form_dan_to_model(_rwfiles, _dan))
# запускаем тест с моделью model_0  и сохрняем веса в "Model_0.ws"
    _nwm.SetModelX(model_0, "Model_0.ws" )
    
    print(" -- грузим веса  --- ")
    model =  _nwm.LoadModelX(model_0, "Model_0.ws", False)

    z1, z2 = _cd.recognizeMultiClass(model, bolezn1)
    print(" По первым симптомам болезнь  {x0}".format(x0=z2))   
    z1, z2 = _cd.recognizeMultiClass(model, bolezn2)
    print(" По вторым симптомам болезнь  {x0}".format(x0=z2))   
