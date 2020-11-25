#pragma once

namespace Param::Calibration
{

/*
inline double offset = 72.;  //ワールド座標系xy平面からメータ平面までの距離

#define IMAGE_NUM (20) // 画像数 
#define PAT_ROW (7)    // パターンの行数 
#define PAT_COL (10)   // パターンの列数 
#define PAT_SIZE (PAT_ROW * PAT_COL)
#define ALL_POINTS (IMAGE_NUM * PAT_SIZE)
//#define CHESS_SIZE (270. * 1.6 / 23.8) // パターン1マスの1辺サイズ[mm] 
#define CHESS_SIZE (24) // パターン1マスの1辺サイズ[mm] 
*/


//for dia
inline double offset = 410.;  //ワールド座標系xy平面からメータ平面までの距離
#define IMAGE_NUM (21)        /* 画像数 */
#define PAT_ROW (6)           /* パターンの行数 */
#define PAT_COL (9)           /* パターンの列数 */
#define PAT_SIZE (PAT_ROW * PAT_COL)
#define ALL_POINTS (IMAGE_NUM * PAT_SIZE)
//#define CHESS_SIZE (270. * 1.6 / 23.8) /* パターン1マスの1辺サイズ[mm] */
#define CHESS_SIZE (38.8) /* パターン1マスの1辺サイズ[mm] */


}  // namespace Param::Calibration
