プログラム内容

- src
 - main.cpp
    値を読み取るプログラム
    実行方法：まとめて実行するスクリプトreadabilityを用いる
    　$ ./readability
 - test.cpp
    位置姿勢推定を行うプログラム
 - prepareTemplate.cpp
  　テンプレート画像を用意するためのプログラム
 - makeMask.cpp
　　テンプレート画像（マスク付き）を用意するためのプログラム
 - setting.cpp
    カメラの内部パラメータを設定するためのプログラム
 - modules/
    関数
 - common/
 - build/
    実行ファイルなど
 - pictures/
    読み取り対象の画像
 - params/
 - externals/
  - AdaptiveThresholding/
  　適応的2値化
  - picojson.h
    jsonファイルの読み書き

