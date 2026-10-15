"""Optional native voting preview; never changes the shared application."""
import sys

sys.dont_write_bytecode=True
from preview import create_app as original_preview, HERE
from voting_model_v3 import VotingMushroom

POLICY_ID='milan_voting_threshold_preview_v3'


def create_app():
    manifest=HERE/'voting_artifacts_v3/voting.json'
    if not manifest.is_file():
        raise RuntimeError('No qualifying voting challenger was packaged. Use the verified v2 preview.py.')
    return original_preview(manifest,policy_id=POLICY_ID,model_reader=VotingMushroom,
        preview_folder=HERE/'.generated/preview_frontend_voting_v3')


if __name__=='__main__':
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=8013)
