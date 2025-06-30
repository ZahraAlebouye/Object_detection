const imageUpload = document.getElementById('image-upload');
const imageCanvas = document.getElementById('image-canvas');
const loader = document.getElementById('loader');
const ctx = imageCanvas.getContext('2d');

let session;

async function loadModel() {
    loader.style.display = 'block';
    try {
        session = await ort.InferenceSession.create('./model.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log('ONNX model loaded successfully.');
    } catch (e) {
        console.error(`Failed to load the ONNX model: ${e}`);
        alert(`Failed to load the model. See console for details.`);
    }
    loader.style.display = 'none';
}

imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    if (!session) {
        alert('Model is not loaded yet. Please wait.');
        return;
    }

    loader.style.display = 'block';

    const image = new Image();
    image.src = URL.createObjectURL(file);
    image.onload = async () => {
        // Preprocess the image
        const { tensor, scale } = preprocess(image);

        // Run inference
        const feeds = { 'images': tensor };
        const results = await session.run(feeds);
        
        // Postprocess and draw
        drawBoundingBoxes(image, results, scale);
        loader.style.display = 'none';
    };
});

function preprocess(image) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const modelWidth = 640;
    const modelHeight = 640;

    canvas.width = modelWidth;
    canvas.height = modelHeight;

    // Calculate scaling factor
    const scale = Math.min(modelWidth / image.width, modelHeight / image.height);
    const scaledWidth = image.width * scale;
    const scaledHeight = image.height * scale;

    // Draw image to canvas with letterboxing
    context.fillStyle = '#000000'; // Or some other color like 114, 114, 114
    context.fillRect(0, 0, modelWidth, modelHeight);
    context.drawImage(image, 0, 0, scaledWidth, scaledHeight);

    const imageData = context.getImageData(0, 0, modelWidth, modelHeight);
    const { data } = imageData;
    const float32Data = new Float32Array(3 * modelWidth * modelHeight);

    // HWC to CHW and normalization
    for (let i = 0; i < modelWidth * modelHeight; i++) {
        float32Data[i] = data[i * 4] / 255.0; // R
        float32Data[i + modelWidth * modelHeight] = data[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * modelWidth * modelHeight] = data[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]);
    return { tensor, scale };
}

function drawBoundingBoxes(image, results, scale) {
    const boxes = results.boxes.data; // Adjust key based on actual model output
    const scores = results.scores.data; // Adjust key based on actual model output
    const labels = results.labels.data; // Adjust key based on actual model output

    // Setup canvas
    imageCanvas.width = image.width;
    imageCanvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    // Draw boxes
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = '#00FF00';

    for (let i = 0; i < scores.length; ++i) {
        if (scores[i] > 0.5) { // Confidence threshold
            const [x1, y1, x2, y2] = [boxes[i*4], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]];
            
            // Rescale boxes to original image size
            const rectX = x1 / scale;
            const rectY = y1 / scale;
            const rectWidth = (x2 - x1) / scale;
            const rectHeight = (y2 - y1) / scale;

            ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);
            ctx.fillText(`Defect: ${scores[i].toFixed(2)}`, rectX, rectY > 10 ? rectY - 5 : 10);
        }
    }
}

// Load the model when the script is loaded
loadModel();
