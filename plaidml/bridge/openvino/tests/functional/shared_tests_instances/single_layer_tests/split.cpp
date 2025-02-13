// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/split.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke_NumSplitsCheck, SplitLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::Values(1),                                        //
                            ::testing::Values(0, 1, 2, 3),                               //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),    //
                            ::testing::Values(std::vector<size_t>({})),                  //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        SplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_splitWithUnusedOutputsTest, SplitLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::Values(5),                                        //
                            ::testing::Values(0),                                        //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),    //
                            ::testing::Values(std::vector<size_t>({0, 3})),              //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        SplitLayerTest::getTestCaseName);
}  // namespace
