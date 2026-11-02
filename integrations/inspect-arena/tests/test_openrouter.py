"""Offline checks of high effort, stable image prefixes and cache-hit evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspect_robots import Scene
from run_inspect import create_agent_policy
from robocurve_inspect.openrouter import OpenRouterTransport
from test_adapter import setup


class OpenRouterTests(unittest.TestCase):
    def test_default_high_effort_and_full_image_prefix_survive_three_decisions(self):
        env,adapter,obs=setup();requests=[];sessions=[]
        def respond(request):
            requests.append(json.loads(request.content));sessions.append(request.headers.get('x-session-id'))
            self.assertEqual(request.extensions['timeout']['read'],300.)
            self.assertEqual(requests[-1]['reasoning'],{'effort':'high'})
            self.assertNotIn('reasoning_effort',requests[-1])
            self.assertNotIn('cache_control',requests[-1])
            cached=0 if len(requests)==1 else 5000
            return httpx.Response(200,json={'id':f'offline-{len(requests)}','model':'google/gemini-4-argon',
                'usage':{'prompt_tokens':6000,'completion_tokens':30,'cost':.01,
                         'prompt_tokens_details':{'cached_tokens':cached}},
                'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'tool_calls':[{'id':f'move-{len(requests)}','type':'function','function':{'name':'move_joints',
                    'arguments':json.dumps({'targets':{'left_gripper_open':.9},'note':'Observe and close slightly.'})}}]}}]})
        policy=create_agent_policy('google/gemini-4-argon',3,transport=httpx.MockTransport(respond),
                                   env={'OPENROUTER_API_KEY':'secret-test-value'})
        self.assertEqual(policy.config.effort,'high')
        self.assertIsNone(policy.config.image_horizon)
        policy.bind(adapter.info);policy.reset(Scene('test','Observe and close slightly'))
        with tempfile.TemporaryDirectory() as directory:
            policy.robocurve_transport.output=Path(directory)
            for _ in range(3):
                for action in policy.act(obs).actions: obs=adapter.step(action).observation
            for first,second in zip(requests,requests[1:]):
                self.assertEqual(first['messages'],second['messages'][:len(first['messages'])])
            self.assertEqual([sum(p.get('type')=='image_url' for m in r['messages']
                                 if isinstance(m.get('content'),list) for p in m['content'])
                              for r in requests],[3,6,9])
            self.assertEqual(len(set(sessions)),1);self.assertTrue(sessions[0])
            summary=json.loads((Path(directory)/'openrouter_usage.json').read_text())
            self.assertTrue(summary['cache_hit_confirmed'])
            self.assertEqual(summary['cached_tokens'],10000)
            text=(Path(directory)/'openrouter_requests.jsonl').read_text()
            self.assertNotIn('secret-test-value',text)
            self.assertNotIn('data:image',text)
            self.assertNotIn('Observe and close',text)

    def test_unreported_cache_usage_is_not_claimed_as_a_hit(self):
        transport=OpenRouterTransport(httpx.MockTransport(lambda r:httpx.Response(200,json={'choices':[]})))
        with tempfile.TemporaryDirectory() as directory:
            transport.output=Path(directory)
            transport.handle_request(httpx.Request('POST','https://openrouter.ai/api/v1/chat/completions',
                                     json={'model':'google/gemini-4-argon','messages':[]}))
            summary=json.loads((Path(directory)/'openrouter_usage.json').read_text())
            self.assertFalse(summary['cache_usage_reported'])
            self.assertFalse(summary['cache_hit_confirmed'])

    def test_explicit_historical_settings_are_available_without_changing_default(self):
        _,adapter,_=setup()
        policy=create_agent_policy('google/gemini-4-argon',1,effort='low',cache_friendly=False,
                                   transport=httpx.MockTransport(lambda r:httpx.Response(200,json={})),
                                   env={'OPENROUTER_API_KEY':'offline-test'})
        self.assertEqual(policy.config.effort,'low')
        self.assertEqual(policy.config.image_horizon,2)
        self.assertIsNone(policy.robocurve_transport.session_id)
