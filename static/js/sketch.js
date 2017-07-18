let reset = false;
let drawing = false;

let outputResult;
let submitButton;

function setup() {
    var canvas = createCanvas(400, 400);
    canvas.mousePressed(startDrawing);
    canvas.mouseReleased(stopDrawing);
    outputResult = createP(' ');
    submitButton = createButton('Classify');
    submitButton.mousePressed(classify);
    background(0);
}

function draw() {
    if (drawing) {
        stroke(255);
        strokeWeight(16);
        line(mouseX, mouseY, pmouseX, pmouseY);
    }
}

function startDrawing() {
    drawing = true;
    if (reset) {
        background(0);
        reset = false;
    }
}

function stopDrawing() {
    drawing = false;
}

function classify() {
    const img = get();
    const base64 = img.canvas.toDataURL();
    const imgClean = base64.replace('data:image/png;base64,', '');
    const data = {
        img: imgClean
    }
    httpPost('/upload', data, success, error);

    function success(reply) {
        const result = JSON.parse(reply);
        console.log(result)
        reset = true;
    }
    function error(reply) {
        console.log(reply)
    }
}