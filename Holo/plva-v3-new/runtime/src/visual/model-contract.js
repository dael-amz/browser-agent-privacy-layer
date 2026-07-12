export const VISUAL_ARTIFACT_CONTRACT = Object.freeze({
  checkpointSha256: "5fcd871b76a6fe456d004ef465ad5cf97616b9f475080711ba0d059a5807a9ad",
  modelSha256: "2ed3f25c9bee375dc1683cf0ffa2044374b6bc35dd89a465a7dce6451ce8b928",
  thresholdArtifactSha256: "c5c9892e0b87d5356645ac70090e22c21f95efd41d2391ad854c45ec87ecea60",
  thresholdProfileSha256: "919da6455fc93ffb8ef929eeb5ded012b102c97a083b421cd0b5226b8be8b90e",
  runtimePolicySha256: "c92fe90f228d7b47e7202fd09efbf77b75e8db5bc0837a6fc145a9774d0c102a",
});

export const VISUAL_RUNTIME_POLICY = Object.freeze({
  schemaVersion: 1,
  inputShape: Object.freeze([1, 3, 640, 640]),
  preprocessing: "single-frame-letterbox-opencv-linear-rgb-fp32-0-1",
  topClassOnly: true,
  classAwareNmsIou: 0.7,
  maxDetections: 300,
  paddingFraction: 0.04,
  adaptiveTiling: false,
});
