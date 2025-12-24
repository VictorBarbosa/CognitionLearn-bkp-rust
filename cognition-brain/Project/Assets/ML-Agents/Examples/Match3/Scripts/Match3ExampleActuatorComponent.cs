using Unity.CognitionLearn;
using Unity.CognitionLearn.Actuators;
using Unity.CognitionLearn.Integrations.Match3;

namespace Unity.CognitionLearnExamples
{
    public class Match3ExampleActuatorComponent : Match3ActuatorComponent
    {
        /// <inheritdoc/>
        public override IActuator[] CreateActuators()
        {
            var board = GetComponent<Match3Board>();
            var seed = RandomSeed == -1 ? gameObject.GetInstanceID() : RandomSeed + 1;
            return new IActuator[] { new Match3ExampleActuator(board, ForceHeuristic, ActuatorName, seed) };
        }
    }
}
