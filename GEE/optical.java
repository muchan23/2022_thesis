var geometry = ee.Geometry.Rectangle([130.69643796118186,32.16909272727556, 130.79943478735373,32.258558460819906]);
 
//geometryが確認できる位置まで地図を移動する
Map.centerObject(geometry)
 
//バンド値を元に画像をつトルゥーカラーとしてレンダリングするための設定
var trueColor = {
  bands:['B4', 'B3', 'B2'],
  min: 0.0,
  max: 0.4,
};
 
//イメージコレクションの中から対象となるイメージを取得
var imageObject = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')   //LANDSAT 8号の画像を指定
                  .filterDate('2020-07-01', '2020-07-30') //期間でフィルタリング
                   .filterMetadata('CLOUD_COVER', 'less_than', 2) //なるべく雲の少ない写真を選ぶ
                  .filterBounds(geometry) //gemetryのエリアが含まれる写真を選択
                  .first() //絞り込んだコレクションの中から最初の１枚を取得
                  .visualize(trueColor) //レンダリング設定を渡す
 
                  
//地図上に衛星画像をオーバーレイレイする
Map.addLayer(imageObject ,null, 'True Color');
 
//イメージをGoogleドライブに出力する
Export.image.toDrive({
  image: imageObject,
  description: 'image',
  scale: 30,
  region: geometry
});
