"""Optional RF feature-selection challenger, including its documented tradeoff."""
import sys
from pathlib import Path

sys.dont_write_bytecode=True
from preview import create_app as original_preview, HERE

POLICY_ID='milan_feature_selection_threshold_preview_v3'


def create_app():
    manifest=HERE/'artifacts_v3/mushroom_rf300.json'
    if not manifest.is_file():
        raise RuntimeError('No useful v3 RF challenger was packaged. Use the verified v2 preview.py.')
    return original_preview(manifest,policy_id=POLICY_ID,
        preview_folder=HERE/'.generated/preview_frontend_v3')


if __name__=='__main__':
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=8012)
