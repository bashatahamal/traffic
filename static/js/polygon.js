prototypefabric.polygon = {
    drawPolygon : function() {
        polygonMode = true;
        pointArray = new Array();
        lineArray = new Array();
        activeLine;
    },
    addPoint : function(options) {
        var random = Math.floor(Math.random() * (max - min + 1)) + min;
        var id = new Date().getTime() + random;
        var circle = new fabric.Circle({
            radius: 5,
            fill: '#ffffff',
            stroke: '#333333',
            strokeWidth: 0.5,
            left: (options.e.layerX/canvas.getZoom()),
            top: (options.e.layerY/canvas.getZoom()),
            selectable: false,
            hasBorders: false,
            hasControls: false,
            originX:'center',
            originY:'center',
            id:id
        });
        if(pointArray.length == 0){
            circle.set({
                fill:'red'
            })
        }
        var points = [(options.e.layerX/canvas.getZoom()),(options.e.layerY/canvas.getZoom()),(options.e.layerX/canvas.getZoom()),(options.e.layerY/canvas.getZoom())];
        line = new fabric.Line(points, {
            strokeWidth: 2,
            fill: '#999999',
            stroke: '#999999',
            class:'line',
            originX:'center',
            originY:'center',
            selectable: false,
            hasBorders: false,
            hasControls: false,
            evented: false
        });
        if(activeShape){
            var pos = canvas.getPointer(options.e);
            var points = activeShape.get("points");
            points.push({
                x: pos.x,
                y: pos.y
            });
            var polygon = new fabric.Polygon(points,{
                stroke:'#333333',
                strokeWidth:1,
                fill: '#cccccc',
                opacity: 0.1,
                selectable: false,
                hasBorders: false,
                hasControls: false,
                evented: false
            });
            canvas.remove(activeShape);
            canvas.add(polygon);
            activeShape = polygon;
            canvas.renderAll();
        }
        else{
            var polyPoint = [{x:(options.e.layerX/canvas.getZoom()),y:(options.e.layerY/canvas.getZoom())}];
            var polygon = new fabric.Polygon(polyPoint,{
                stroke:'#333333',
                strokeWidth:1,
                fill: '#cccccc',
                opacity: 0.1,
                selectable: false,
                hasBorders: false,
                hasControls: false,
                evented: false
            });
            activeShape = polygon;
            canvas.add(polygon);
        }
        activeLine = line;

        pointArray.push(circle);
        lineArray.push(line);

        canvas.add(line);
        canvas.add(circle);
        canvas.selection = false;
    },
    generatePolygon : function(pointArray){
        var points = new Array();
        $.each(pointArray,function(index,point){
            points.push({
                x:point.left,
                y:point.top
            });
            canvas.remove(point);
        });
        $.each(lineArray,function(index,line){
            canvas.remove(line);
        });
        canvas.remove(activeShape).remove(activeLine);
        var polygon = new fabric.Polygon(points,{
            stroke:'#333333',
            strokeWidth:0.5,
            fill: 'red',
            opacity: 0.35,
            hasBorders: false,
            hasControls: false,
            selectable: false, 
            evented: false
        });
        p = []
        for (x = 0; x < points.length; x++) {
			p.push(points[x])
		}
        PolyCoord.push(p)
        DirectList.push(direction)
        // console.log(points)
        // console.log(p)
//         AddMidText(points, PolyCoord.length + '; ' + 'UP')
        cmid = get_polygon_centroid(points)
        canvas.add(new fabric.Text(PolyCoord.length + '; ' + direction, { 
                    fontFamily: 'Delicious_500', 
                    left: cmid['x'], 
                    top: cmid['y'],
                    fontSize: 20,
                    textAlign: 'left'
        }));
        canvas.add(polygon);

        activeLine = null;
        activeShape = null;
        polygonMode = false;
        canvas.selection = true;
    },
    drawLine : function(pointArray){
//         console.log('inside_script_drawline')
        var points = new Array();
        $.each(pointArray,function(index,point){
            points.push({
                x:point.left,
                y:point.top
            });
            canvas.remove(point);
        });
        $.each(lineArray,function(index,line){
            canvas.remove(line);
        });
        canvas.remove(activeShape).remove(activeLine);
//         console.log(points)
        p = [points[0]['x'], points[0]['y'],
             points[1]['x'], points[1]['y']]
        LineCoord.push(p)
        line = new fabric.Line(p, {
            strokeWidth: 3,
//             fill: '#999999',
//             stroke: '#999999',
            fill: '#00FF00',
            stroke: '#00FF00',
            class:'line',
            originX:'center',
            originY:'center',
            selectable: false,
            hasBorders: false,
            hasControls: false,
            evented: false
        });
        canvas.add(line);

        activeLine = null;
        activeShape = null;
        polygonMode = false;
        canvas.selection = true;
    },
    

};

function get_polygon_centroid(pts) {
   var first = pts[0], last = pts[pts.length-1];
   if (first.x != last.x || first.y != last.y) pts.push(first);
   var twicearea=0,
   x=0, y=0,
   nPts = pts.length,
   p1, p2, f;
   for ( var i=0, j=nPts-1 ; i<nPts ; j=i++ ) {
      p1 = pts[i]; p2 = pts[j];
      f = p1.x*p2.y - p2.x*p1.y;
      twicearea += f;          
      x += ( p1.x + p2.x ) * f;
      y += ( p1.y + p2.y ) * f;
   }
   f = twicearea * 3;
   return { x:x/f, y:y/f };
}


function AddMidText(coord, text) {
    var c_mid = center(coord)
    ctx = document.getElementById('c').getContext('2d')
    ctx.textAlign = 'center'
    ctx.font = '20px Arial'
    ctx.fillText(text, c_mid[0], c_mid[1])
}
    
    
    var center = function (arr)
{
    var minX, maxX, minY, maxY;
    for (var i = 0; i < arr.length; i++)
    {
        minX = (arr[i]['x'] < minX || minX == null) ? arr[i]['x'] : minX;
        maxX = (arr[i]['x'] > maxX || maxX == null) ? arr[i]['x'] : maxX;
        minY = (arr[i]['y'] < minY || minY == null) ? arr[i]['y'] : minY;
        maxY = (arr[i]['y'] > maxY || maxY == null) ? arr[i]['y'] : maxY;
    }
    return [(minX + maxX) / 2, (minY + maxY) / 2];
}
