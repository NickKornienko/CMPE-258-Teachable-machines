let model;
const labels = ["DownDog","Goddess","Plank","Tree","Warrior"]

async function loadModel() {
  model = await tf.loadLayersModel("model/model.json");
  console.log("Model loaded!");
}

async function predict(imageElement) {
  console.log("Predicting...");
  if (!model) {
    console.warn("Model not loaded, please wait...");
    return;
  }

  const tensor = tf.browser
    .fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])
    .toFloat();

  const batched = tensor.expandDims(0);
  const normalized = batched.div(255);
  const prediction = await model.predict(normalized).data();

  displayPrediction(prediction);
}

function displayPrediction(predictionArray) {
  const highestProbabilityIndex = predictionArray.indexOf(
    Math.max(...predictionArray)
  );
  const predictedClass = labels[highestProbabilityIndex];
  document.getElementById(
    "predictionResult"
  ).innerText = `Predicted class: ${predictedClass} with probability ${predictionArray[highestProbabilityIndex]}`;
}

function handleUpload() {
  console.log("Uploading image...");
  const imageUpload = document.getElementById("imageUpload");
  const reader = new FileReader();

  reader.onload = async (e) => {
    const img = new Image();
    img.src = e.target.result;
    img.onload = () => predict(img);
  };

  reader.readAsDataURL(imageUpload.files[0]);
}

loadModel();
