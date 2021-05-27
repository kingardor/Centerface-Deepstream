#include <cstring>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

#include "nvdsinfer_custom_impl.h"

float HEIGHT = 544.0;
float WIDTH = 960.0;

float PROB_THRESHOLD = 0.2;
float NMS_THRESHOLD = 0.45;

typedef struct {
  float score;
  float x1;
  float x2;
  float y1;
  float y2;
  float landmarks[10];
} Faces;

std::vector<std::vector<std::vector<float>>>
reshape(float *buffer, int channels, int height, int width) {
  std::vector<std::vector<std::vector<float>>> reshaped;
  for (int c = 0; c < channels; c++) {
    std::vector<std::vector<float>> channel_vector;
    for (int h = 0; h < height; h++) {
      std::vector<float> width_vector;
      for(int w = 0; w < width; w++) {
        width_vector.emplace_back(buffer[(c * height + h) * width + w]);
      }
      channel_vector.emplace_back(width_vector);
    }
    reshaped.emplace_back(channel_vector);
  }
  return reshaped;
}

std::vector<std::array<int, 2>>
getFilteredCoordinates(std::vector<std::vector<float>>heatmap,
                       int hHeight, int hWidth, float probthreshold) {
  std::vector<std::array<int, 2>> filteredHeatmap;
  for(int h = 0; h < hHeight; h++) {
    for(int w = 0; w < hWidth; w++) {
      if (heatmap[h][w] >= probthreshold) {
        std::array<int, 2>coords = {h, w};
        filteredHeatmap.emplace_back(coords);
      }
    }
  }
  return filteredHeatmap;
}

std::vector<Faces>
nms (std::vector<Faces>& input, float nmsthreshold) {
  std::vector<Faces> faceData;
  std::sort(input.begin(), input.end(),
    [](const Faces&a, const Faces& b)
    {
      return a.score > b.score;
    });

  int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  for(int i = 0; i < box_num; i++) {
    if(merged[i]) {
      continue;
    }

    faceData.emplace_back(input[i]);

    float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

    for(int j = i + 1; j < box_num; j++) {
      if(merged[j]) {
        continue;
      }

      float inner_x0 = std::max(input[i].x1, input[j].x1);
      float inner_y0 = std::max(input[i].y1, input[j].y1);

      float inner_x1 = std::min(input[i].x2, input[j].x2);
      float inner_y1 = std::min(input[i].y2, input[j].y2);

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

      if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score >= nmsthreshold)
				merged[j] = true;
    }
  }
  return faceData;
}

void
squareBox(std::vector<Faces>& faces) {
	float w=0, h=0, maxSize=0;
	float cenx, ceny;
	for (unsigned int i = 0; i < faces.size(); i++) {
		w = faces[i].x2 - faces[i].x1;
		h = faces[i].y2 - faces[i].y1;

		maxSize = std::max(w, h);
		cenx = faces[i].x1 + w / 2;
		ceny = faces[i].y1 + h / 2;

		faces[i].x1 = std::max(cenx - maxSize / 2, 0.f);
		faces[i].y1 = std::max(ceny - maxSize/ 2, 0.f);
		faces[i].x2 = std::min(cenx + maxSize / 2, WIDTH - 1.f);
		faces[i].y2 = std::min(ceny + maxSize / 2, HEIGHT - 1.f);
	}
}

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomCenterFace (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomCenterFace (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList) {
  static NvDsInferDimsCHW HeatmapDims;
  static NvDsInferDimsCHW ScaleDims;
  static NvDsInferDimsCHW OffsetDims;
  static NvDsInferDimsCHW LandmarksDims;

  static int HeatmapLayerIndex = -1;
  static int ScaleLayerIndex = -1;
  static int OffsetLayerIndex = -1;
  static int LandmarksLayerIndex = -1;
  static bool classMismatchWarn = false;
  int numClassesToParse;

  // Find Heatmap Layer
  if (HeatmapLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "537") == 0) {
        HeatmapLayerIndex = i;
        getDimsCHWFromDims(HeatmapDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (HeatmapLayerIndex == -1) {
    std::cerr << "Could not find heatmap layer buffer while parsing" << std::endl;
    return false;
    }
  }

  // Find Scale Layer
  if (ScaleLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "538") == 0) {
        ScaleLayerIndex = i;
        getDimsCHWFromDims(ScaleDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (ScaleLayerIndex == -1) {
    std::cerr << "Could not find scale layer buffer while parsing" << std::endl;
    return false;
    }
  }

  // Find Offset Layer
  if (OffsetLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "539") == 0) {
        OffsetLayerIndex = i;
        getDimsCHWFromDims(OffsetDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (OffsetLayerIndex == -1) {
    std::cerr << "Could not find offset layer buffer while parsing" << std::endl;
    return false;
    }
  }

  // Find Landmarks Layer
  if (LandmarksLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "540") == 0) {
        LandmarksLayerIndex = i;
        getDimsCHWFromDims(LandmarksDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (LandmarksLayerIndex == -1) {
    std::cerr << "Could not find landmarks layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Warn in case of mismatch in number of classes */
  if (!classMismatchWarn) {
    if (HeatmapDims.c != detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        HeatmapDims.c << std::endl;
    }
    classMismatchWarn = true;
  }

  // Calculate the number of classes to parse
  numClassesToParse = std::min(HeatmapDims.c, detectionParams.numClassesConfigured);
  // Access all buffers
  float *outputHeatmapBuf = (float *) outputLayersInfo[HeatmapLayerIndex].buffer;
  int heatmapC = HeatmapDims.c;
  int heatmapH = HeatmapDims.h;
  int heatmapW = HeatmapDims.w;

  std::vector<std::vector<std::vector<float>>> reshapedHeatmapBuf =
    reshape(outputHeatmapBuf, heatmapC, heatmapH, heatmapW);

  float *outputScaleBuf = (float *) outputLayersInfo[ScaleLayerIndex].buffer;
  int scaleC = ScaleDims.c;
  int scaleH = ScaleDims.h;
  int scaleW = ScaleDims.w;
  std::vector<std::vector<std::vector<float>>> reshapedScaleBuf =
    reshape(outputScaleBuf, scaleC, scaleH, scaleW);

  float *outputOffsetBuf = (float *) outputLayersInfo[OffsetLayerIndex].buffer;
  int offsetC = OffsetDims.c;
  int offsetH = OffsetDims.h;
  int offsetW = OffsetDims.w;
  std::vector<std::vector<std::vector<float>>> reshapedOffsetBuf =
    reshape(outputOffsetBuf, offsetC, offsetH, offsetW);

  float *outputLandmarksBuf = (float *) outputLayersInfo[LandmarksLayerIndex].buffer;
  int landmarksC = LandmarksDims.c;
  int landmarksH = LandmarksDims.h;
  int landmarksW = LandmarksDims.w;
  std::vector<std::vector<std::vector<float>>> reshapedLandmarksBuf =
    reshape(outputLandmarksBuf, landmarksC, landmarksH, landmarksW);

  // Squeeze Heatmap Buffer
  std::vector<std::vector<float>> squeezedHeatmap =
    reshapedHeatmapBuf[0];

  // Divide and squeeze Scales Buffer
  std::vector<std::vector<std::vector<float>>> split_scale0(
    reshapedScaleBuf.begin(),
    reshapedScaleBuf.begin() + reshapedScaleBuf.size() / 2);
  std::vector<std::vector<std::vector<float>>> split_scale1(
    reshapedScaleBuf.begin() + reshapedScaleBuf.size() / 2,
    reshapedScaleBuf.end());

  std::vector<std::vector<float>> scale0 = split_scale0[0];
  std::vector<std::vector<float>> scale1 = split_scale1[0];

  // Divide and squeeze Offset Buffer
  std::vector<std::vector<std::vector<float>>> split_offset0(
    reshapedOffsetBuf.begin(),
    reshapedOffsetBuf.begin() + reshapedOffsetBuf.size() / 2);
  std::vector<std::vector<std::vector<float>>> split_offset1(
    reshapedOffsetBuf.begin() + reshapedOffsetBuf.size() / 2,
    reshapedOffsetBuf.end());
  std::vector<std::vector<float>> offset0 = split_offset0[0];
  std::vector<std::vector<float>> offset1 = split_offset1[0];

  // Get coordinates above threshold
  std::vector<std::array<int, 2>> filteredHeatmapCoords =
    getFilteredCoordinates(squeezedHeatmap, heatmapH, heatmapW, PROB_THRESHOLD);

  if(filteredHeatmapCoords.size() <= 0) {
    return true;
  }

  // Generate coordinates and landmarks
  std::vector<Faces> rawFaces;
  for(unsigned int i = 0; i < filteredHeatmapCoords.size(); i++) {
    std::array<int, 2> coords = filteredHeatmapCoords[i];
    float s0 = std::exp(scale0[coords[0]][coords[1]]) * 4;
    float s1 = std::exp(scale1[coords[0]][coords[1]]) * 4;
    float o0 = offset0[coords[0]][coords[1]];
    float o1 = offset1[coords[0]][coords[1]];
    float score = squeezedHeatmap[coords[0]][coords[1]];

    float x1 = 0.0, y1 = 0.0,
          x2 = 0.0, y2 = 0.0;
    x1 = std::max(0.0, (coords[1] + o1 + 0.5) * 4 - s1 / 2);
    y1 = std::max(0.0,(coords[0] + o0 + 0.5) * 4 - s0 / 2);
    x1 = std::min(x1, WIDTH);
    y1 = std::min(y1, HEIGHT);
    x2 = std::min(x1 + s1, WIDTH);
    y2 = std::min(y1 + s0, HEIGHT);

    Faces rawFace;
    rawFace.score = score;
    rawFace.x1 = x1;
    rawFace.y1 = y1;
    rawFace.x2 = x2;
    rawFace.y2 = y2;

    for(int j = 0; j < 5; j++) {
      rawFace.landmarks[2*j] =
        reshapedLandmarksBuf[j * 2 + 1][coords[0]][coords[1]] * s1 + x1;
      rawFace.landmarks[2*j+1] =
        reshapedLandmarksBuf[j * 2][coords[0]][coords[1]] * s0 + y1;
    }
    rawFaces.emplace_back(rawFace);
  }

  // NMS
  std::vector<Faces> faceData =
    nms(rawFaces, NMS_THRESHOLD);

  // Square Boxes
  // squareBox(faceData);

  // Pass values to DS
  for (int c = 0; c < numClassesToParse; c++) {
    for(unsigned int i = 0; i < faceData.size(); i++) {
      NvDsInferObjectDetectionInfo object;
      int x = 0, y = 0, w = 0, h = 0;
      x = faceData[i].x1;
      y = faceData[i].y1;
      w = faceData[i].x2 - faceData[i].x1;
      h = faceData[i].y2 - faceData[i].y1;

      object.classId = c;
      object.detectionConfidence = faceData[i].score;
      object.left = x;
      object.top = y;
      object.width = w;
      object.height = h;
      objectList.emplace_back(object);
    }
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomCenterFace);
