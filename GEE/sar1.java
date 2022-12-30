var geometry = ee.Geometry.Rectangle([130.69643796118186,32.16909272727556, 130.79943478735373,32.258558460819906]);
 
// Load the Sentinel-1 ImageCollection.
var imgVV = ee.ImageCollection('COPERNICUS/S1_GRD')
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'));
var desc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
var asc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));

// Filter by date and geometry, create multiband image and clip to geometry
var ascChange = ee.ImageCollection(asc.filterDate('2020-07-01', '2020-09-01')).filterBounds(geometry).toBands().toFloat().clip(geometry)

//geometryが確認できる位置まで地図を移動する
Map.centerObject(geometry)
Map.addLayer(ascChange, {min: -25, max: 5}, 'Multi-T Mean ASC', true);

Export.image.toDrive({
  image: ascChange,
  description:'Ascendente_CMillor',
  fileNamePrefix: 'scenedate_orbitProperties_pass_GRD',
  scale: 10,
  region: geometry})
