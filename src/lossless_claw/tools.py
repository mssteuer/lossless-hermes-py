"""LCM tool implementations.

Provides lcm_grep, lcm_describe, and lcm_expand tools for the context engine.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from .db.connection import get_database
from .store.conversation import ConversationStore
from .store.summary import SummaryStore
from .retrieval import RetrievalEngine, SearchQuery, SearchResult


logger = logging.getLogger(__name__)


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return tool schemas for LCM tools."""
    return [
        {
            "name": "lcm_grep",
            "description": "Search through conversation history and summaries using full-text search or regex patterns. Useful for finding past discussions, decisions, or specific technical details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Can be keywords, phrases, or regex pattern depending on mode."
                    },
                    "mode": {
                        "type": "string", 
                        "enum": ["full_text", "regex"],
                        "default": "full_text",
                        "description": "Search mode: 'full_text' for FTS5 search, 'regex' for pattern matching."
                    },
                    "include_messages": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to search message content."
                    },
                    "include_summaries": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Whether to search summary content."
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of results to return."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "lcm_describe",
            "description": "Get detailed information about the current conversation's LCM state, including summary statistics, compaction history, and DAG structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_stats": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include summary statistics by depth and type."
                    },
                    "include_recent": {
                        "type": "boolean",
                        "default": True, 
                        "description": "Include information about recent activity."
                    }
                },
                "required": []
            }
        },
        {
            "name": "lcm_expand",
            "description": "Retrieve and expand specific content from the conversation history. Can get full message content, summary details, or related context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_type": {
                        "type": "string",
                        "enum": ["message", "summary", "related"],
                        "description": "Type of content to expand: 'message' for full message content, 'summary' for summary details, 'related' for finding related content."
                    },
                    "target_id": {
                        "type": "string",
                        "description": "ID of the target (message_id for messages, summary_id for summaries)."
                    },
                    "query": {
                        "type": "string", 
                        "description": "Search query for 'related' target_type to find related content."
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1, 
                        "maximum": 50,
                        "description": "Maximum number of results for 'related' queries."
                    }
                },
                "required": ["target_type"]
            }
        }
    ]


class LcmTools:
    """LCM tool implementations."""
    
    def __init__(self, conversation_id: int):
        self.conversation_id = conversation_id
        
        # Initialize stores
        db = get_database()
        if not db:
            raise RuntimeError("LCM database not initialized")
        
        self.conversation_store = ConversationStore(db)
        self.summary_store = SummaryStore(db)
        self.retrieval_engine = RetrievalEngine(
            self.conversation_store,
            self.summary_store
        )
    
    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """Handle a tool call and return JSON response."""
        try:
            if name == "lcm_grep":
                return self._handle_lcm_grep(args)
            elif name == "lcm_describe":
                return self._handle_lcm_describe(args)
            elif name == "lcm_expand":
                return self._handle_lcm_expand(args)
            else:
                return json.dumps({
                    "error": f"Unknown LCM tool: {name}"
                })
        except Exception as e:
            logger.error(f"LCM tool error ({name}): {e}")
            return json.dumps({
                "error": f"Tool execution failed: {str(e)}"
            })
    
    def _handle_lcm_grep(self, args: Dict[str, Any]) -> str:
        """Handle lcm_grep tool call."""
        query_text = args.get("query", "").strip()
        if not query_text:
            return json.dumps({
                "error": "Query parameter is required"
            })
        
        search_query = SearchQuery(
            query=query_text,
            conversation_id=self.conversation_id,
            include_messages=args.get("include_messages", True),
            include_summaries=args.get("include_summaries", True),
            limit=args.get("limit", 20),
            mode=args.get("mode", "full_text")
        )
        
        results = self.retrieval_engine.search(search_query)
        
        # Format results for response
        formatted_results = []
        for result in results:
            formatted_result = {
                "type": result.type,
                "id": str(result.id),
                "snippet": result.snippet,
                "metadata": result.metadata or {}
            }
            
            if result.relevance_score is not None:
                formatted_result["relevance_score"] = result.relevance_score
            
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "query": query_text,
            "mode": search_query.mode,
            "results_count": len(results),
            "results": formatted_results
        })
    
    def _handle_lcm_describe(self, args: Dict[str, Any]) -> str:
        """Handle lcm_describe tool call."""
        include_stats = args.get("include_stats", True)
        include_recent = args.get("include_recent", True)
        
        response = {
            "conversation_id": self.conversation_id
        }
        
        # Get basic conversation info
        conversation = self.conversation_store.get_conversation_by_session(
            f"conv_{self.conversation_id}"  # This is a placeholder - needs proper session mapping
        )
        
        if conversation:
            response["conversation"] = {
                "active": conversation.active,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "title": conversation.title,
                "bootstrapped_at": conversation.bootstrapped_at.isoformat() if conversation.bootstrapped_at else None
            }
        
        if include_stats:
            # Get summary statistics
            stats = self.summary_store.get_summary_depth_stats(self.conversation_id)
            response["summary_stats"] = stats
            
            # Get message count
            messages = self.conversation_store.get_messages_by_conversation(self.conversation_id)
            response["message_count"] = len(messages)
            response["total_message_tokens"] = sum(msg.token_count for msg in messages)
        
        if include_recent:
            # Get recent summaries
            recent_summaries = self.summary_store.get_summaries_by_conversation(
                self.conversation_id
            )
            recent_summaries.sort(key=lambda s: s.created_at, reverse=True)
            
            response["recent_summaries"] = []
            for summary in recent_summaries[:5]:  # Last 5 summaries
                response["recent_summaries"].append({
                    "summary_id": summary.summary_id,
                    "kind": summary.kind,
                    "depth": summary.depth,
                    "token_count": summary.token_count,
                    "descendant_count": summary.descendant_count,
                    "created_at": summary.created_at.isoformat(),
                    "model": summary.model
                })
        
        return json.dumps(response, indent=2)
    
    def _handle_lcm_expand(self, args: Dict[str, Any]) -> str:
        """Handle lcm_expand tool call."""
        target_type = args.get("target_type")
        target_id = args.get("target_id")
        query = args.get("query", "")
        limit = args.get("limit", 10)
        
        if target_type == "message":
            if not target_id:
                return json.dumps({
                    "error": "target_id is required for message expansion"
                })
            
            try:
                message_id = int(target_id)
            except (ValueError, TypeError):
                return json.dumps({
                    "error": "target_id must be a valid message ID"
                })
            
            # Get full message content
            messages = self.conversation_store.get_messages_by_conversation(self.conversation_id)
            target_message = None
            
            for msg in messages:
                if msg.message_id == message_id:
                    target_message = msg
                    break
            
            if not target_message:
                return json.dumps({
                    "error": f"Message {message_id} not found"
                })
            
            # Get message parts if available
            parts = self.conversation_store.get_message_parts(message_id)
            
            response = {
                "message_id": message_id,
                "role": target_message.role,
                "content": target_message.content,
                "token_count": target_message.token_count,
                "identity_hash": target_message.identity_hash,
                "created_at": target_message.created_at.isoformat(),
                "seq": target_message.seq
            }
            
            if parts:
                response["parts"] = []
                for part in parts:
                    response["parts"].append({
                        "part_id": part.part_id,
                        "part_type": part.part_type,
                        "ordinal": part.ordinal,
                        "text_content": part.text_content,
                        "tool_call_id": part.tool_call_id,
                        "tool_name": part.tool_name,
                        "tool_input": part.tool_input,
                        "tool_output": part.tool_output,
                        "metadata": part.metadata
                    })
            
            return json.dumps(response, indent=2)
        
        elif target_type == "summary":
            if not target_id:
                return json.dumps({
                    "error": "target_id is required for summary expansion"
                })
            
            summary = self.summary_store.get_summary(target_id)
            if not summary:
                return json.dumps({
                    "error": f"Summary {target_id} not found"
                })
            
            # Get linked messages
            linked_message_ids = self.summary_store.get_summary_messages(target_id)
            
            # Get parent/child relationships
            parents = self.summary_store.get_summary_parents(target_id)
            children = self.summary_store.get_summary_children(target_id)
            
            response = {
                "summary_id": summary.summary_id,
                "kind": summary.kind,
                "depth": summary.depth,
                "content": summary.content,
                "token_count": summary.token_count,
                "descendant_count": summary.descendant_count,
                "descendant_token_count": summary.descendant_token_count,
                "source_message_token_count": summary.source_message_token_count,
                "file_ids": summary.file_ids,
                "model": summary.model,
                "created_at": summary.created_at.isoformat(),
                "earliest_at": summary.earliest_at.isoformat() if summary.earliest_at else None,
                "latest_at": summary.latest_at.isoformat() if summary.latest_at else None,
                "linked_messages": linked_message_ids,
                "parent_summaries": parents,
                "child_summaries": children
            }
            
            return json.dumps(response, indent=2)
        
        elif target_type == "related":
            if not query:
                return json.dumps({
                    "error": "query is required for related content search"
                })
            
            # Search for related content
            search_query = SearchQuery(
                query=query,
                conversation_id=self.conversation_id,
                include_messages=True,
                include_summaries=True,
                limit=limit,
                mode="full_text"
            )
            
            results = self.retrieval_engine.search(search_query)
            
            # Also find similar conversations
            similar_conversations = self.retrieval_engine.search_similar_conversations(
                query=query,
                exclude_conversation_id=self.conversation_id,
                limit=5
            )
            
            formatted_results = []
            for result in results:
                formatted_result = {
                    "type": result.type,
                    "id": str(result.id),
                    "snippet": result.snippet,
                    "metadata": result.metadata or {}
                }
                
                if result.relevance_score is not None:
                    formatted_result["relevance_score"] = result.relevance_score
                
                formatted_results.append(formatted_result)
            
            response = {
                "query": query,
                "current_conversation": self.conversation_id,
                "related_content": formatted_results,
                "similar_conversations": similar_conversations
            }
            
            return json.dumps(response, indent=2)
        
        else:
            return json.dumps({
                "error": f"Unknown target_type: {target_type}. Must be 'message', 'summary', or 'related'."
            })