var min = 99;
var max = 999999;
var polygonMode = true;
var pointArray = new Array();
var lineArray = new Array();
var activeLine;
var activeShape = false;
var canvas
var PolyCoord = []
var LineCoord = []
var DirectList = []
var direction = ''
var drawing_line = false
$(window).load(function () {
// $("#add_line").on("click", function (value) {
    // function create_fabric(h,w) {
    // prototypefabric.initCanvas(h,w);
    prototypefabric.initCanvas();
    $('#add_poly_up').click(function () {
        prototypefabric.polygon.drawPolygon();
        direction = 'UP';
    });
    $('#add_poly_down').click(function () {
        prototypefabric.polygon.drawPolygon();
        direction = 'DOWN';
    });
    $('#add_poly_left').click(function () {
        prototypefabric.polygon.drawPolygon();
        direction = 'LEFT';
    });
    $('#add_poly_right').click(function () {
        prototypefabric.polygon.drawPolygon();
        direction = 'RIGHT';
    });
    $('#add_line').click(function () {
        prototypefabric.polygon.drawPolygon();
        //         canvas.clear()
        drawing_line = true
    });
    // }
});

// var centerCanvasPosition = function(canvas){
//     canvas.style.left = window.innerWidth / 2 - canvas.width / 2 + 'px';
//     canvas.style.top =window.innerHeight / 2 - canvas.height / 2 + 'px';
//   };

var prototypefabric = new function () {
    this.initCanvas = function () {
        // /this.initCanvas = function (h,w) {
        canvas = window._canvas = new fabric.Canvas('canvas2');
        // canvas.setWidth(w);
        // canvas.setHeight(h);
        // centerCanvasPosition(document.getElementById('canvas2'))
        // canvas.setWidth($(window).width());
        // canvas.setHeight($(window).height() - $('#nav-bar').height());
        // canvas.selection = false;

        canvas.on('mouse:down', function (options) {
            //             console.log('From PT Poly')
            if (options.target && options.target.id == pointArray[0].id && drawing_line == false) {
                prototypefabric.polygon.generatePolygon(pointArray);
                display_regA()
            }
            if (polygonMode && (direction != '' || drawing_line == true)) {
                prototypefabric.polygon.addPoint(options);
            }
        });
        canvas.on('mouse:up', function (options) {
            if (drawing_line == true && pointArray.length > 1) {
                //                 console.log('inside_drawing_line')
                prototypefabric.polygon.drawLine(pointArray)
                drawing_line = false
            }
        });
        canvas.on('mouse:move', function (options) {
            if (activeLine && activeLine.class == "line") {
                var pointer = canvas.getPointer(options.e);
                activeLine.set({ x2: pointer.x, y2: pointer.y });

                var points = activeShape.get("points");
                points[pointArray.length] = {
                    x: pointer.x,
                    y: pointer.y
                }
                activeShape.set({
                    points: points
                });
                canvas.renderAll();
            }
            canvas.renderAll();
        });
    };
};
