"""Exercise the installed SDK retry loop with offline HTTP failures."""
import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspect_robots import Scene
from test_adapter import setup
from run_inspect import create_agent_policy


class RequestRetryTests(TestCase):
    def policy(self, adapter, handler):
        policy=create_agent_policy('google/gemini-4-argon',1,transport=httpx.MockTransport(handler),
                                   env={'OPENROUTER_API_KEY':'offline-test'})
        policy.bind(adapter.info)
        policy.reset(Scene('test','Close the left gripper slightly'))
        return policy

    def test_timeout_then_503_then_success_does_not_replay_motion(self):
        env,adapter,obs=setup();requests=[]
        def respond(request):
            requests.append(request.content)
            if len(requests)==1: raise httpx.ReadTimeout('offline timeout',request=request)
            if len(requests)==2: return httpx.Response(503,text='offline unavailable')
            return httpx.Response(200,json={'choices':[{'finish_reason':'tool_calls','message':{
                'role':'assistant','content':None,'tool_calls':[{'id':'retry-move','type':'function',
                'function':{'name':'move_joints','arguments':json.dumps({
                    'targets':{'left_gripper_open':.9},'note':'Close left gripper slightly.'})}}]}}]})
        policy=self.policy(adapter,respond)
        with patch('inspect_robots_agent._llm.time.sleep') as sleep:
            chunk=policy.act(obs)
        self.assertEqual(len(requests),3)
        self.assertEqual(len(set(requests)),1)
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[2.,4.])
        self.assertEqual(len(env.native_calls),0)
        for action in chunk.actions: adapter.step(action)
        self.assertEqual(len(env.native_calls),len(chunk.actions))
        self.assertAlmostEqual(adapter.observe().state['gripper'][0],.9)

    def test_persistent_timeout_stops_after_three_attempts_without_motion(self):
        env,adapter,obs=setup();requests=[]
        def respond(request):
            requests.append(request)
            raise httpx.ReadTimeout('offline timeout',request=request)
        policy=self.policy(adapter,respond)
        with patch('inspect_robots_agent._llm.time.sleep'), self.assertRaisesRegex(Exception,'after 3 attempts'):
            policy.act(obs)
        self.assertEqual(len(requests),3)
        self.assertEqual(len(env.native_calls),0)

    def test_authentication_error_is_not_retried(self):
        env,adapter,obs=setup();requests=[]
        def respond(request):
            requests.append(request)
            return httpx.Response(401,text='offline invalid key')
        policy=self.policy(adapter,respond)
        with patch('inspect_robots_agent._llm.time.sleep') as sleep, self.assertRaisesRegex(Exception,'401'):
            policy.act(obs)
        self.assertEqual(len(requests),1)
        sleep.assert_not_called()
        self.assertEqual(len(env.native_calls),0)
